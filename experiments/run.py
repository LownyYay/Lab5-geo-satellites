"""One experiment = one config. Primary estimate: the supplied validation set (held out, never trained on).
Second estimate (--cv-fold k): stratified 5-fold CV on train, model trained on 4/5 of train, evaluated on the
held-out fold. Results go to experiments/results/<name>.json (+ val probabilities for paired bootstrap)."""
import argparse
import json
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from common import (accuracy_se, build_model, load_split, predict_proba, save_result, seed_everything, train)

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--arch", default="resnet18")
p.add_argument("--size", type=int, default=128)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--backbone-lr-mult", type=float, default=0.1)
p.add_argument("--wd", type=float, default=1e-4)
p.add_argument("--freeze", default=None, help="stem|layer1|layer2|layer3 (ResNet only)")
p.add_argument("--dropout", type=float, default=0.2)
p.add_argument("--ls", type=float, default=0.0, help="label smoothing")
p.add_argument("--corrupt", type=float, default=0.0, help="probability of corruption augmentation")
p.add_argument("--crop", type=float, default=0.0, help="min area of random resized crop (0 = off)")
p.add_argument("--jitter", type=float, default=0.0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--cv-fold", type=int, default=-1)
p.add_argument("--no-pretrained", action="store_true")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--threads", type=int, default=0)
args = p.parse_args()
if args.threads:
    torch.set_num_threads(args.threads)

seed_everything(args.seed)
tr_x, tr_y, _ = load_split("train")
va_x, va_y, _ = load_split("val")
eval_sets = {"val": (va_x, va_y)}
if args.cv_fold >= 0:
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(tr_x, tr_y))
    fit_idx, hold_idx = folds[args.cv_fold]
    eval_sets["cv"] = (tr_x[hold_idx], tr_y[hold_idx])
    tr_x, tr_y = tr_x[fit_idx], tr_y[fit_idx]

aug = {"corrupt": args.corrupt, "jitter": args.jitter}
if args.crop > 0:
    aug["crop"] = (args.crop, 1.0)
cfg = dict(arch=args.arch, size=args.size, epochs=args.epochs, bs=args.bs, lr=args.lr,
           backbone_lr_mult=args.backbone_lr_mult, wd=args.wd, label_smoothing=args.ls, seed=args.seed,
           aug=aug, freeze=args.freeze, dropout=args.dropout, cv_fold=args.cv_fold,
           pretrained=not args.no_pretrained, device=args.device, eval_every=max(1, args.epochs // 5))

model = build_model(args.arch, pretrained=not args.no_pretrained, freeze_until=args.freeze, dropout=args.dropout)
t0 = time.time()
history = train(model, tr_x, tr_y, cfg, args.device, eval_sets=eval_sets)
train_s = time.time() - t0

t0 = time.time()
val_p = predict_proba(model, va_x, args.size, args.device)
infer_s = time.time() - t0
val_p_tta = predict_proba(model, va_x, args.size, args.device, tta=True)
acc = float((val_p.argmax(1) == va_y).mean())
acc_tta = float((val_p_tta.argmax(1) == va_y).mean())
res = {"name": args.name, "cfg": cfg, "val_acc": acc, "val_se": accuracy_se(acc, len(va_y)),
       "val_acc_tta": acc_tta, "train_s": train_s, "val_infer_s": infer_s, "history": history}
if args.cv_fold >= 0:
    cv_p = predict_proba(model, eval_sets["cv"][0], args.size, args.device)
    res["cv_acc"] = float((cv_p.argmax(1) == eval_sets["cv"][1]).mean())
save_result(args.name, res, probs=np.stack([val_p, val_p_tta]))
print(json.dumps({k: v for k, v in res.items() if k not in ("history", "cfg")}, indent=1))
