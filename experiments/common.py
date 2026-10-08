"""Shared code for experiments: data cache, corruption augmentations, models, training loop, evaluation."""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision.io import decode_jpeg, encode_jpeg

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = Path(os.environ.get("LAB5_CACHE", ROOT / "experiments" / "cache"))
RESULTS = ROOT / "experiments" / "results"
NUM_CLASSES = 20
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------- data
def load_split(split):
    """uint8 array (N, 256, 256, 3) and labels; cached as .npy after the first read."""
    table = pd.read_csv(DATA / f"{split}.csv", dtype={"id": str})
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{split}_256.npy"
    if path.exists():
        images = np.load(path)
    else:
        images = np.stack([np.asarray(Image.open(DATA / split / i).convert("RGB")) for i in table.id])
        np.save(path, images)
    labels = table.label.to_numpy() if "label" in table else np.full(len(table), -1)
    return np.asarray(images), labels, table.id.tolist()


# ---------------------------------------------------------------- augmentations (batched, on device)
def dihedral(x):
    """Random element of the dihedral group D4 per sample: 4 rotations x optional flip."""
    out = torch.empty_like(x)
    k = torch.randint(0, 4, (x.shape[0],))
    flip = torch.rand(x.shape[0]) < 0.5
    for i in range(x.shape[0]):
        xi = torch.rot90(x[i], int(k[i]), dims=(1, 2))
        out[i] = xi.flip(2) if flip[i] else xi
    return out


def gaussian_kernel(sigma, device):
    radius = max(1, int(3 * sigma + 0.5))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def blur(img, sigma):
    """Separable Gaussian blur of one image (3, H, W) in [0, 1]."""
    k = gaussian_kernel(sigma, img.device)
    r = (len(k) - 1) // 2
    x = img.unsqueeze(0)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)
    return x[0]


def jpeg(img, quality):
    u8 = (img.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
    return (decode_jpeg(encode_jpeg(u8, quality=int(quality))).float() / 255).to(img.device)


def corrupt(x, p, gen=None):
    """Simulate the degradations found in val (EDA): blur / low resolution, JPEG, grayscale, noise, low contrast.
    x: (B, 3, H, W) float in [0, 1]. Each image is corrupted with probability p by 1-2 random operations."""
    out = x.clone()
    H = x.shape[-1]
    for i in range(x.shape[0]):
        if random.random() >= p:
            continue
        ops = random.sample(["blur", "lowres", "jpeg", "gray", "noise", "contrast"], k=random.choice([1, 1, 2]))
        img = out[i]
        for op in ops:
            if op == "blur":
                img = blur(img, random.uniform(0.8, 3.0))
            elif op == "lowres":
                s = random.randint(H // 6, H // 2)
                img = F.interpolate(F.interpolate(img[None], size=s, mode="bilinear", antialias=True, align_corners=False),
                                    size=H, mode="bilinear", align_corners=False)[0]
            elif op == "jpeg":
                img = jpeg(img, random.randint(8, 40))
            elif op == "gray":
                g = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2])[None]
                img = g.expand(3, -1, -1).clone()
            elif op == "noise":
                img = img + torch.randn_like(img) * random.uniform(0.02, 0.08)
            elif op == "contrast":
                c = random.uniform(0.4, 0.8)
                img = (img - img.mean()) * c + img.mean() + random.uniform(-0.1, 0.1)
        out[i] = img.clamp(0, 1)
    return out


def color_jitter(x, strength=0.2):
    b = 1 + (torch.rand(x.shape[0], 1, 1, 1, device=x.device) * 2 - 1) * strength
    c = 1 + (torch.rand(x.shape[0], 1, 1, 1, device=x.device) * 2 - 1) * strength
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - mean) * c + mean) * b


def random_resized_crop(x, scale=(0.6, 1.0)):
    """Per-sample random square crop (zoom-in) resized back to the input size."""
    B, _, H, W = x.shape
    out = torch.empty_like(x)
    for i in range(B):
        s = int(H * random.uniform(*scale) ** 0.5 + 0.5)
        top, left = random.randint(0, H - s), random.randint(0, W - s)
        out[i] = F.interpolate(x[i:i + 1, :, top:top + s, left:left + s], size=(H, W), mode="bilinear",
                               antialias=True, align_corners=False)[0]
    return out


def prepare_batch(u8, size, device, train, aug):
    """uint8 (B, 256, 256, 3) numpy -> normalized float tensor (B, 3, size, size)."""
    x = torch.from_numpy(np.ascontiguousarray(u8)).to(device).permute(0, 3, 1, 2).float() / 255
    if train:
        x = dihedral(x)
        if aug.get("crop"):
            x = random_resized_crop(x, aug["crop"])
        if aug.get("corrupt", 0) > 0:
            x = corrupt(x, aug["corrupt"])
        if aug.get("jitter", 0) > 0:
            x = color_jitter(x, aug["jitter"]).clamp(0, 1)
    if size != x.shape[-1]:
        x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


# ---------------------------------------------------------------- models
BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "googlenet": (models.googlenet, models.GoogLeNet_Weights.IMAGENET1K_V1),
    "alexnet": (models.alexnet, models.AlexNet_Weights.IMAGENET1K_V1),
    "vgg11_bn": (models.vgg11_bn, models.VGG11_BN_Weights.IMAGENET1K_V1),
}


def build_model(arch, pretrained=True, freeze_until=None, dropout=0.2):
    """Pretrained backbone (classifier removed) + our own head, combined with nn.Sequential.
    freeze_until: name of the last frozen ResNet stage ('stem', 'layer1', 'layer2', 'layer3') or None."""
    ctor, weights = BACKBONES[arch]
    backbone = ctor(weights=weights if pretrained else None)
    if arch.startswith("resnet") or arch == "googlenet":
        n_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
    else:  # alexnet / vgg: keep the first FC layers, drop the 1000-class layer
        n_features = backbone.classifier[6].in_features
        backbone.classifier[6] = nn.Identity()
    if freeze_until is not None and arch.startswith("resnet"):
        order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
        stop = {"stem": 2, "layer1": 3, "layer2": 4, "layer3": 5}[freeze_until]
        for name in order[:stop]:
            for p in getattr(backbone, name).parameters():
                p.requires_grad = False
    head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, NUM_CLASSES))
    return nn.Sequential(backbone, head)


def set_frozen_bn_eval(model):
    """Frozen BN layers keep ImageNet running statistics (do not update them in train mode)."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


# ---------------------------------------------------------------- training / evaluation
@torch.no_grad()
def predict_proba(model, images, size, device, batch=128, tta=False):
    model.eval()
    probs = []
    for s in range(0, len(images), batch):
        x = prepare_batch(images[s:s + batch], size, device, train=False, aug={})
        if tta:
            views = [torch.rot90(x, k, dims=(2, 3)) for k in range(4)]
            views += [v.flip(3) for v in views]
            p = torch.stack([model(v).float().softmax(1) for v in views]).mean(0)
        else:
            p = model(x).float().softmax(1)
        probs.append(p.cpu())
    return torch.cat(probs).numpy()


def train(model, images, labels, cfg, device, eval_sets=None, log=print):
    """Train with AdamW + warmup/cosine schedule. Lower LR for the pretrained backbone (discriminative LR).
    Returns per-epoch history; no checkpoint selection on validation (the final epoch is used)."""
    seed_everything(cfg["seed"])
    model = model.to(device)
    backbone, head = model[0], model[1]
    params = [
        {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": cfg["lr"] * cfg.get("backbone_lr_mult", 0.1)},
        {"params": head.parameters(), "lr": cfg["lr"]},
    ]
    opt = torch.optim.AdamW(params, weight_decay=cfg.get("wd", 1e-4))
    steps_per_epoch = int(np.ceil(len(images) / cfg["bs"]))
    total = cfg["epochs"] * steps_per_epoch
    warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    use_amp = device == "cuda" and cfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(cfg["seed"])
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); set_frozen_bn_eval(model)
        t0, tot, correct = time.time(), 0.0, 0
        order = rng.permutation(len(images))
        for s in range(0, len(order), cfg["bs"]):
            idx = np.sort(order[s:s + cfg["bs"]])
            x = prepare_batch(images[idx], cfg["size"], device, train=True, aug=cfg.get("aug", {}))
            y = torch.from_numpy(labels[idx]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(idx)
            correct += (out.argmax(1) == y).sum().item()
        rec = {"epoch": epoch, "train_loss": tot / len(images), "train_acc": correct / len(images),
               "epoch_s": time.time() - t0}
        for name, (ev_images, ev_labels) in (eval_sets or {}).items():
            if epoch == cfg["epochs"] or epoch % cfg.get("eval_every", 1) == 0:
                p = predict_proba(model, ev_images, cfg["size"], device)
                rec[f"{name}_acc"] = float((p.argmax(1) == ev_labels).mean())
                rec[f"{name}_loss"] = float(-np.log(p[np.arange(len(ev_labels)), ev_labels] + 1e-12).mean())
        history.append(rec)
        log(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
    return history


def accuracy_se(acc, n):
    return float(np.sqrt(acc * (1 - acc) / n))


def save_result(name, payload, probs=None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=1))
    if probs is not None:
        np.save(RESULTS / f"{name}_valprobs.npy", probs)
