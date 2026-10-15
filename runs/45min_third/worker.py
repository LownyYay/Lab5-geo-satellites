"""Обучение части ансамбля на ОДНОЙ видеокарте + предсказание test.

Ноутбук Kaggle_45min.ipynb запускает этот файл отдельным процессом на каждую GPU:
    CUDA_VISIBLE_DEVICES=0 python worker.py --gpu 0 --models r152 r50  --t0 <таймер ноутбука> ...
    CUDA_VISIBLE_DEVICES=1 python worker.py --gpu 1 --models r101 inc3 --t0 <таймер ноутбука> ...
Процессы ничего не обменивают во время обучения: каждый обучает свои модели на всех 5698 размеченных снимках
(train + val) по рецепту из Team_final.ipynb и сохраняет вероятности для test. Ансамбль собирает ноутбук.

Отличия от Team_final.ipynb (рецепт обучения тот же):
  * только этап full (обучение на train + val), без holdout: оценка на val и выбор размера TTA уже сделаны;
  * cudnn.benchmark = True: cuDNN сам выбирает самые быстрые свёртки (вход всегда одного размера);
  * AdamW(fused=True) и счётчики loss / accuracy на GPU: нет синхронизации CPU-GPU на каждом шаге;
  * «предохранитель» времени (time_guard): после первой эпохи (разогрев) известна реальная скорость; если
    20 эпох не успевают к BUDGET минут от старта таймера, число эпох уменьшается (косинус строится на новое
    число эпох). При скорости как в наших прогонах на Colab он не срабатывает.
"""
import argparse
import gc
import io
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=True, help="номер GPU (для подписей; саму GPU задаёт CUDA_VISIBLE_DEVICES)")
parser.add_argument("--models", nargs="+", required=True, help="ключи из PRESETS, в порядке обучения")
parser.add_argument("--t0", type=float, required=True, help="time.time() в ноутбуке в момент старта таймера")
parser.add_argument("--budget", type=float, default=40.0, help="к этой минуте от t0 обучение должно закончиться")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--min-epochs", type=int, default=10)
parser.add_argument("--tta", type=int, default=8, choices=[1, 2, 4, 8], help="число вариантов снимка в TTA")
parser.add_argument("--workers", type=int, default=2, help="процессы DataLoader (аугментации на CPU)")
parser.add_argument("--benchmark", type=int, default=1, help="1: cudnn.benchmark; 0: детерминированный cuDNN")
parser.add_argument("--arrays", required=True, help="папка с trainval_images.npy, trainval_labels.npy, test_images.npy")
parser.add_argument("--out", required=True, help="папка для результатов")
args = parser.parse_args()

T0, BUDGET, OUT, ARRAYS = args.t0, args.budget, Path(args.out), Path(args.arrays)
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"
NUM_CLASSES = 20
torch.set_num_threads(1)      # главный процесс почти не считает на CPU; ядра CPU нужны процессам аугментации


def log(message):
    print(f"[GPU {args.gpu} | {(time.time() - T0) / 60:5.1f} мин] {message}", flush=True)


# ================================================================================ настройки (как в Team_final)
COMMON = dict(lr=2e-4, head_lr_mult=10, weight_decay=0.01, label_smoothing=0.1, warmup_epochs=1,
              crop_scale=(0.5, 1.0), corruption_p=0.5, jitter=0.3, dropout=0.2, epochs=args.epochs, batch=32,
              seed=43)
PRESETS = {
    # test_sizes — размер тестового снимка, выбранный на val в holdout-прогонах (раздел 9 Team_final.ipynb)
    "r50": dict(arch="resnet50", size=256, test_sizes=[320]),
    "r101": dict(arch="resnet101", size=256, test_sizes=[320]),
    "r152": dict(arch="resnet152", size=256, test_sizes=[320]),
    "inc3": dict(arch="inception_v3", size=299, test_sizes=[363]),
    "inc3_363": dict(arch="inception_v3", size=363, test_sizes=[427]),
}
# минут на эпоху full-обучения (5698 снимков) в наших прогонах на Colab T4 — для прогноза времени
COLAB_EPOCH_MIN = {"r50": 0.502, "r101": 0.759, "r152": 1.067, "inc3": 0.642, "inc3_363": 0.922}


def get_cfg(name):
    return {**COMMON, **PRESETS[name]}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(args.benchmark)
    torch.backends.cudnn.deterministic = not args.benchmark


# ================================================================================ аугментации (как в Team_final)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
GRAY_WEIGHTS = torch.tensor([0.299, 0.587, 0.114], device=DEVICE).view(1, 3, 1, 1)


class RandomCorruption:
    # С вероятностью p применяет 1 или 2 случайных искажения, похожих на найденные в val-снимках.
    OPS = ("blur", "low_res", "jpeg", "gray", "noise", "low_contrast")

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        for op in random.sample(self.OPS, k=random.choice([1, 1, 2])):
            img = getattr(self, op)(img)
        return img

    def blur(self, img):                     # гауссово размытие, sigma 0.8...3 px
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.8, 3.0)))

    def low_res(self, img):                  # уменьшаем в 2...6 раз и увеличиваем обратно
        size = img.width
        small = max(8, int(size * random.uniform(1 / 6, 1 / 2)))
        return img.resize((small, small), Image.BILINEAR).resize((size, size), Image.BILINEAR)

    def jpeg(self, img):                     # сильное JPEG-сжатие, качество 8...40
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=random.randint(8, 40))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def gray(self, img):                     # оттенки серого, 3 канала сохраняются
        return img.convert("L").convert("RGB")

    def noise(self, img):                    # гауссов шум, std 5...20 уровней яркости
        rng = np.random.default_rng(random.getrandbits(32))
        pixels = np.asarray(img, dtype=np.float32)
        pixels = pixels + rng.normal(0, random.uniform(5, 20), pixels.shape)
        return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))

    def low_contrast(self, img):             # контраст x0.4...0.8
        return ImageEnhance.Contrast(img).enhance(random.uniform(0.4, 0.8))


class RandomRot90:
    ROTATIONS = (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270)

    def __call__(self, img):
        rotation = random.choice(self.ROTATIONS)
        return img if rotation is None else img.transpose(rotation)


class ToUint8Tensor:
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)


def make_train_transform(size, cfg):
    return transforms.Compose([
        RandomCorruption(p=cfg["corruption_p"]),
        transforms.RandomResizedCrop(size, scale=cfg["crop_scale"]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRot90(),
        ToUint8Tensor(),
    ])


def gpu_color_jitter(x, strength, generator):
    def factor():
        return 1 + (torch.rand(x.shape[0], 1, 1, 1, device=x.device, generator=generator) * 2 - 1) * strength
    x = x * factor()                                                      # яркость
    mean = (x * GRAY_WEIGHTS).sum(1, keepdim=True).mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * factor() + mean                                      # контраст
    gray = (x * GRAY_WEIGHTS).sum(1, keepdim=True)
    x = (x - gray) * factor() + gray                                      # насыщенность
    return x.clamp(0, 1)


def to_model_input(x_u8, size=None, jitter=0.0, generator=None):
    x = x_u8.to(DEVICE, non_blocking=True).float().div_(255)
    if jitter > 0:
        x = gpu_color_jitter(x, jitter, generator)
    if size is not None and x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.contiguous(memory_format=torch.channels_last)


class ArrayDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(np.array(self.images[i]))), int(self.labels[i])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_train_loader(images, labels, cfg):
    generator = torch.Generator().manual_seed(cfg["seed"])
    return DataLoader(ArrayDataset(images, labels, make_train_transform(cfg["size"], cfg)),
                      batch_size=cfg["batch"], shuffle=True, drop_last=True, num_workers=args.workers,
                      worker_init_fn=seed_worker, generator=generator, pin_memory=True,
                      persistent_workers=args.workers > 0,
                      # fork (как на Colab): процессы аугментации получают массив снимков без копирования
                      multiprocessing_context="fork" if args.workers > 0 else None)


# ================================================================================ модели (как в Team_final)
BACKBONES = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2),
    "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V2),
    "inception_v3": (models.inception_v3, models.Inception_V3_Weights.IMAGENET1K_V1),
}


def build_model(cfg):
    constructor, weights = BACKBONES[cfg["arch"]]
    body = constructor(weights=weights)      # веса уже скачаны ноутбуком в кэш torchvision
    if cfg["arch"] == "inception_v3":        # transform_input=True ставят сами веса IMAGENET1K_V1
        body.aux_logits, body.AuxLogits = False, None
        body.dropout = nn.Identity()
    n_features = body.fc.in_features
    body.fc = nn.Identity()
    head = nn.Sequential(nn.Dropout(cfg["dropout"]), nn.Linear(n_features, NUM_CLASSES))
    return nn.Sequential(body, head)


# ================================================================================ обучение
def lr_factor(step, warmup_steps, total_steps):
    # разогрев: lr линейно растёт первую эпоху; затем косинус до 0 к последнему шагу
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))


def time_guard(name, epoch1_min, epochs, later):
    """После первой эпохи: успеваем ли все оставшиеся эпохи этой модели и следующих моделей к BUDGET минуте?
    Если нет — уменьшаем число эпох этой модели (следующие проверяются так же после своей первой эпохи)."""
    now = (time.time() - T0) / 60
    speed = epoch1_min / COLAB_EPOCH_MIN[name]                       # >1 — медленнее, чем на Colab
    rest_this = (epochs - 1) * epoch1_min
    rest_later = speed * sum(COLAB_EPOCH_MIN[m] * args.epochs for m in later)
    projected = now + rest_this + rest_later
    log(f"[guard] {name}: 1-я эпоха {epoch1_min:.2f} мин ({speed:.2f}x от Colab); прогноз конца обучения на этой GPU "
        f"{projected:.1f} мин (бюджет {BUDGET:.0f})")
    if projected <= BUDGET:
        return epochs
    share = max(0.0, (BUDGET - now) / (rest_this + rest_later))
    new_epochs = max(args.min_epochs, 1 + int(share * (epochs - 1)))
    log(f"[guard] !!! {name}: не успеваем — эпох {epochs} -> {new_epochs}")
    return new_epochs


def train_model(name, images, labels, later):
    cfg = get_cfg(name)
    seed_everything(cfg["seed"])
    model = build_model(cfg).to(DEVICE, memory_format=torch.channels_last)
    loader = make_train_loader(images, labels, cfg)
    steps_per_epoch = len(loader)
    epochs = cfg["epochs"]
    optimizer = torch.optim.AdamW([
        {"params": model[0].parameters(), "lr": cfg["lr"]},
        {"params": model[1].parameters(), "lr": cfg["lr"] * cfg["head_lr_mult"]},
    ], weight_decay=cfg["weight_decay"], fused=True)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda")
    jitter_generator = torch.Generator(device=DEVICE).manual_seed(cfg["seed"])

    history, step, epoch = [], 0, 0
    while epoch < epochs:                    # epochs может уменьшиться после 1-й эпохи (time_guard)
        epoch += 1
        model.train()
        start = time.time()
        total_steps = epochs * steps_per_epoch
        loss_sum = torch.zeros((), device=DEVICE)
        correct = torch.zeros((), device=DEVICE, dtype=torch.long)
        seen = 0
        for x_u8, y in loader:
            factor = lr_factor(step, warmup_steps, total_steps)
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * factor
            x = to_model_input(x_u8, jitter=cfg["jitter"], generator=jitter_generator)
            y = y.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(x)
                loss = criterion(outputs, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach().float() * len(y)       # счётчики остаются на GPU: без синхронизации
            correct += (outputs.argmax(1) == y).sum()
            seen += len(y)
            step += 1
        torch.cuda.synchronize()
        minutes = (time.time() - start) / 60
        history.append({"epoch": epoch, "train_loss": loss_sum.item() / seen, "train_acc": correct.item() / seen,
                        "lr_factor": factor, "minutes": minutes})
        log(f"{name} эпоха {epoch}/{epochs}: loss {history[-1]['train_loss']:.4f}, "
            f"acc {history[-1]['train_acc']:.4f}, {minutes:.2f} мин")
        if epoch == 1:
            epochs = time_guard(name, minutes, epochs, later)
    return model, pd.DataFrame(history), epochs


# ================================================================================ предсказание
# варианты снимка для TTA: (число поворотов на 90°, отражение). 8 = все повороты и отражения (группа D4)
VIEW_SETS = {8: [(k, flip) for flip in (False, True) for k in range(4)],
             4: [(0, False), (2, False), (0, True), (2, True)],
             2: [(0, False), (0, True)],
             1: [(0, False)]}


@torch.no_grad()
def predict_proba(model, images, size, views, batch=128):
    model.eval()
    result = []
    for start in range(0, len(images), batch):
        x = to_model_input(torch.from_numpy(np.array(images[start:start + batch])).permute(0, 3, 1, 2), size)
        probs = 0
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for k, flip in views:
                v = x.flip(3) if flip else x
                v = torch.rot90(v, k, dims=(2, 3)).contiguous(memory_format=torch.channels_last)
                probs = probs + model(v).float().softmax(1)
        result.append((probs / len(views)).cpu())
    return torch.cat(result).numpy()


# ================================================================================ main
def save_timing(timing):
    # сначала во временный файл, потом переименование: ноутбук никогда не прочитает файл наполовину
    tmp = OUT / f"timing_gpu{args.gpu}.json.tmp"
    tmp.write_text(json.dumps(timing, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(OUT / f"timing_gpu{args.gpu}.json")


def main():
    log(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | модели {args.models} | "
        f"DataLoader workers {args.workers} | cudnn.benchmark {bool(args.benchmark)}")
    images = np.load(ARRAYS / "trainval_images.npy", mmap_mode="r")      # общий файл, в память не копируется
    labels = np.load(ARRAYS / "trainval_labels.npy")
    timing = {"gpu": args.gpu, "models": args.models, "t0": T0, "t_worker_start": time.time(), "per_model": {}}

    # 1) обучение всех моделей этой GPU; test в это время не трогаем
    trained = {}
    for i, name in enumerate(args.models):
        start = time.time()
        model, history, epochs_used = train_model(name, images, labels, later=args.models[i + 1:])
        history.to_csv(OUT / f"history_{name}.csv", index=False)
        timing["per_model"][name] = {"train_min": (time.time() - start) / 60, "epochs": epochs_used}
        trained[name] = model.cpu()              # обученная модель ждёт предсказания в памяти CPU
        del model
        gc.collect()
        torch.cuda.empty_cache()
        log(f"{name} обучена за {timing['per_model'][name]['train_min']:.1f} мин ({epochs_used} эпох)")
    timing["t_train_end"] = time.time()
    save_timing(timing)                          # ноутбук видит, что обучение на этой GPU закончено
    log(f"ОБУЧЕНИЕ НА ЭТОЙ GPU ЗАКОНЧЕНО: {(timing['t_train_end'] - T0) / 60:.1f} мин от старта таймера")

    # 2) предсказание test обученными моделями (test используется только здесь)
    test_images = np.load(ARRAYS / "test_images.npy", mmap_mode="r")
    for name in args.models:
        start = time.time()
        model = trained.pop(name).to(DEVICE, memory_format=torch.channels_last)
        sizes = get_cfg(name)["test_sizes"]
        probs = np.mean([predict_proba(model, test_images, s, VIEW_SETS[args.tta]) for s in sizes], axis=0)
        np.save(OUT / f"test_probs_{name}.npy", probs.astype(np.float32))
        timing["per_model"][name]["predict_min"] = (time.time() - start) / 60
        log(f"{name}: test предсказан ({sizes} px, TTA x{args.tta}) за "
            f"{timing['per_model'][name]['predict_min']:.1f} мин")
        del model
        torch.cuda.empty_cache()
    timing["t_pred_end"] = time.time()
    save_timing(timing)
    log("готово")


if __name__ == "__main__":
    main()
