"""CPU speed benchmark. Paste into ONE cell of a Kaggle notebook (Accelerator: None / CPU, Internet: off is fine)
and send back the printed table. Random weights are used, so no downloads are needed. Runtime: ~6-8 minutes.
"""
import os
import platform
import time

import torch
import torch.nn as nn
from torchvision import models

torch.manual_seed(0)
print("torch", torch.__version__, "| threads", torch.get_num_threads(), "| cpu_count", os.cpu_count(),
      "|", platform.processor() or platform.machine())
try:
    print([l for l in open("/proc/cpuinfo") if l.startswith("model name")][0].strip())
except OSError:
    pass


def bench_infer(name, ctor, size, n_images=256, batch=64):
    model = ctor(weights=None).eval()
    x = torch.randn(batch, 3, size, size)
    with torch.inference_mode():
        model(x)                                    # warm-up
        t = time.perf_counter()
        for _ in range(n_images // batch):
            model(x)
        dt = time.perf_counter() - t
    ms = 1000 * dt / n_images
    print(f"infer  {name:10s} {size:3d}px  {ms:6.2f} ms/img  -> 8299 test imgs: {ms * 8299 / 1000:6.1f} s")


def bench_train(name, ctor, size, trainable_prefixes=None, n_images=128, batch=32):
    model = ctor(weights=None).train()
    for n, p in model.named_parameters():
        p.requires_grad = trainable_prefixes is None or n.startswith(tuple(trainable_prefixes))
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    x, y = torch.randn(batch, 3, size, size), torch.randint(0, 20, (batch,))
    loss_fn = nn.CrossEntropyLoss()
    def step():
        opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()
    step()
    t = time.perf_counter()
    for _ in range(n_images // batch):
        step()
    dt = time.perf_counter() - t
    ms = 1000 * dt / n_images
    tag = "all" if trainable_prefixes is None else "+".join(trainable_prefixes)
    print(f"train  {name:10s} {size:3d}px  [{tag:14s}] {ms:6.2f} ms/img -> epoch of 5698 imgs: {ms * 5698 / 1000 / 60:5.2f} min")


for size in (112, 128, 160, 224):
    bench_infer("resnet18", models.resnet18, size)
for size in (128, 160):
    bench_infer("resnet34", models.resnet34, size)
    bench_infer("resnet50", models.resnet50, size)
    bench_infer("googlenet", lambda weights: models.googlenet(weights=weights, aux_logits=False, init_weights=False), size)
bench_infer("alexnet", models.alexnet, 224)
bench_infer("vgg11_bn", models.vgg11_bn, 128)

bench_train("resnet18", models.resnet18, 128)
bench_train("resnet18", models.resnet18, 128, ["layer3", "layer4", "fc"])
bench_train("resnet18", models.resnet18, 128, ["layer4", "fc"])
bench_train("resnet18", models.resnet18, 160)
bench_train("resnet34", models.resnet34, 128, ["layer4", "fc"])
bench_train("resnet50", models.resnet50, 128, ["layer4", "fc"])

# JPEG decode speed on the competition images, if attached
import glob
from PIL import Image
files = sorted(glob.glob("/kaggle/input/**/test/*.jpg", recursive=True))[:500]
if files:
    t = time.perf_counter()
    for f in files:
        with Image.open(f) as im:
            im.convert("RGB").load()
    ms = 1000 * (time.perf_counter() - t) / len(files)
    print(f"jpeg decode 256px: {ms:.2f} ms/img -> 8299 imgs: {ms * 8299 / 1000:.1f} s")
