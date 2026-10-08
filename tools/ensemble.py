"""Local ensemble of Colab runs (numpy only, no torch).

Reads runs/<run>/{summary.json,val_probs.npy,test_probs.npy}. The val probabilities come from holdout models
(trained on train only); the test probabilities come from full models (train+val).
Prints per-run and ensemble val accuracy (overall / clean / degraded) and the test agreement between runs.
Writes runs/submission_ensemble_<runs>.csv.

usage: python tools/ensemble.py [run1 run2 ...]
default: every run whose val accuracy is within 2 SE of the best one (the rule fixed in advance in the notebook)
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

LAB = Path(__file__).resolve().parents[1]
DATA, RUNS = LAB / "data", LAB / "runs"
val_table = pd.read_csv(DATA / "val.csv", dtype={"id": str})
test_table = pd.read_csv(DATA / "test.csv", dtype={"id": str})
val_labels = val_table.label.to_numpy()


def image_stats(rgb_u8):
    """Same statistics as in the notebooks (section 'Какие val-снимки искажены')."""
    rgb = rgb_u8.astype(np.float32)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    laplacian = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    jumps = np.abs(np.diff(gray, axis=1))
    blockiness = jumps[:, 7::8].mean() / (np.delete(jumps, np.s_[7::8], axis=1).mean() + 1e-9)
    saturation = np.asarray(Image.fromarray(rgb_u8).convert("HSV"), dtype=np.float32)[..., 1].mean()
    return {"sharpness": float(laplacian.var()), "contrast": float(gray.std()),
            "saturation": float(saturation), "blockiness": float(blockiness)}


def val_degraded_mask():
    cache = RUNS / "val_degraded.npy"
    if cache.exists():
        return np.load(cache)
    rows = []
    for split in ("train", "val"):
        table = pd.read_csv(DATA / f"{split}.csv", dtype={"id": str})
        for image_id, label in zip(table.id, table.label):
            with Image.open(DATA / split / image_id) as img:
                rows.append({"split": split, "label": label, **image_stats(np.asarray(img.convert("RGB")))})
    stats = pd.DataFrame(rows)
    train_stats = stats[stats.split == "train"]
    low = train_stats.groupby("label")[["sharpness", "contrast"]].quantile(0.01).add_suffix("_q01")
    high = train_stats.groupby("label")[["sharpness", "blockiness"]].quantile(0.99).add_suffix("_q99")
    s = stats.join(low, on="label").join(high, on="label")
    degraded = ((s.sharpness < s.sharpness_q01) | (s.sharpness > s.sharpness_q99) | (s.blockiness > s.blockiness_q99)
                | (s.saturation < 8) | (s.contrast < s.contrast_q01)).to_numpy()
    mask = degraded[stats.split.to_numpy() == "val"]
    np.save(cache, mask)
    return mask


def report(probs, degraded):
    correct = probs.argmax(1) == val_labels
    acc = correct.mean()
    return {"val acc": acc, "± SE": np.sqrt(acc * (1 - acc) / len(correct)), "errors": int((~correct).sum()),
            "clean": correct[~degraded].mean(), "degraded": correct[degraded].mean()}


available = sorted(p.name for p in RUNS.iterdir() if p.is_dir() and not p.name.endswith("_quick")
                   and (p / "val_probs.npy").exists() and (p / "test_probs.npy").exists())
degraded = val_degraded_mask()
print(f"degraded val images: {degraded.sum()} / {len(degraded)}")

val = {r: np.load(RUNS / r / "val_probs.npy") for r in available}
test = {r: np.load(RUNS / r / "test_probs.npy") for r in available}
rows = []
for r in available:
    info = json.loads((RUNS / r / "summary.json").read_text(encoding="utf-8"))
    rows.append({"run": r, "test option": info.get("test_option"), **report(val[r], degraded)})
singles = pd.DataFrame(rows).set_index("run")

if sys.argv[1:]:
    runs = sys.argv[1:]
else:   # pre-registered rule (same as the notebook): within 2 SE of the best single model
    best = singles["val acc"].idxmax()
    threshold = singles.loc[best, "val acc"] - 2 * singles.loc[best, "± SE"]
    runs = [r for r in available if singles.loc[r, "val acc"] >= threshold]
    print(f"rule: val acc >= {threshold:.4f}; excluded: {sorted(set(available) - set(runs)) or 'none'}")
rows.append({"run": "ensemble: " + " + ".join(runs), "test option": "mean",
             **report(np.mean([val[r] for r in runs], axis=0), degraded)})
for r in runs if len(runs) > 2 else []:   # leave-one-out: how much each model adds
    rest = [q for q in runs if q != r]
    rows.append({"run": f"  without {r}", "test option": "mean",
                 **report(np.mean([val[q] for q in rest], axis=0), degraded)})
print(pd.DataFrame(rows).round(4).to_string(index=False))

print("\nagreement of test predictions between runs:")
for a, b in itertools.combinations(runs, 2):
    print(f"  {a} vs {b}: {(test[a].argmax(1) == test[b].argmax(1)).mean():.4f}")
val_pred = {r: val[r].argmax(1) for r in runs}
for a, b in itertools.combinations(runs, 2):
    only_a = ((val_pred[a] == val_labels) & (val_pred[b] != val_labels)).sum()
    only_b = ((val_pred[b] == val_labels) & (val_pred[a] != val_labels)).sum()
    print(f"  val: only {a} right {only_a}, only {b} right {only_b}")

ensemble = np.mean([test[r] for r in runs], axis=0)
submission = pd.DataFrame({"id": test_table.id, "label": ensemble.argmax(1)})
assert submission.id.tolist() == test_table.id.tolist() and submission.label.between(0, 19).all()
out = RUNS / f"submission_ensemble_{'_'.join(runs)}.csv"
submission.to_csv(out, index=False)
shares = submission.label.value_counts(normalize=True).sort_index() * 100
print(f"\n{out.name}: {len(submission)} rows, class shares {shares.min():.1f}...{shares.max():.1f}%")
