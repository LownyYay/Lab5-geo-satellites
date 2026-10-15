"""Candidate submissions from the EXISTING test_probs.npy (equal-weight mean of probabilities).

The candidates are fixed in advance by a principle (what fits into 45 min on 2 x T4 or 1 x T4), not by ranking
hundreds of subsets on val (val cannot tell them apart: see out/subsets.csv). From test probabilities we only take
argmax labels and count how many labels differ between candidates.

usage: python make_candidates.py   -> ../candidates/*.csv and ../candidates/candidates.csv (table)
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

LAB = Path(__file__).resolve().parents[2]
RUNS, DATA = LAB / "runs", LAB / "data"
OUT = Path(__file__).resolve().parents[1] / "candidates"
OUT.mkdir(exist_ok=True)

val_table = pd.read_csv(DATA / "val.csv", dtype={"id": str})
test_table = pd.read_csv(DATA / "test.csv", dtype={"id": str})
y = val_table.label.to_numpy()
degraded = np.load(RUNS / "val_degraded.npy")
POOL = ["r50", "r101", "r152", "inc3", "r50_320", "r101_320", "inc3_363", "r50_mix", "inc3_363_mix"]
val = {m: np.load(RUNS / m / "val_probs.npy") for m in POOL}
test = {m: np.load(RUNS / m / "test_probs.npy") for m in POOL}
full_min = {m: json.loads((RUNS / m / "summary.json").read_text(encoding="utf-8"))["full_train_min"] for m in POOL}


def mean(ms, d):
    return np.mean([d[m] for m in ms], axis=0)


def best_split(ms):
    """2 GPUs: split the models so that the slower GPU finishes as early as possible"""
    best = None
    for r in range(len(ms) + 1):
        for g0 in itertools.combinations(ms, r):
            g1 = [m for m in ms if m not in g0]
            t = max(sum(full_min[m] for m in g0), sum(full_min[m] for m in g1))
            if best is None or t < best[0]:
                best = (t, list(g0), g1)
    return best


REF = {"A": ["r50", "r101", "r152", "inc3"],
       "7": ["r50", "r101", "r152", "inc3", "r50_320", "r101_320", "inc3_363"],
       "r50+r101": ["r50", "r101"]}
PUBLIC = {"A": 0.96678, "7": 0.96678, "r50+r101": 0.96506}
CANDIDATES = {
    # name: (models, hardware, why)
    "A_2gpu_r50_r101_r152_inc3": (REF["A"], "2xT4",
                                  "reference = the 4-model ensemble already on the LB (0.96678); main plan reproduces it"),
    "B_2gpu_r50_r101_r152_inc3-363": (["r50", "r101", "r152", "inc3_363"], "2xT4",
                                      "A with inc3 replaced by Inception trained at 363 px"),
    "C_2gpu_r50_r152_inc3_inc3-363": (["r50", "r152", "inc3", "inc3_363"], "2xT4",
                                      "2 ResNets + 2 Inceptions (Res-Inc pairs disagree most on test)"),
    "S1_1gpu_r50_inc3": (["r50", "inc3"], "1xT4", "cheapest ResNet + Inception pair (~23 min)"),
    "S2_1gpu_r50_inc3-363": (["r50", "inc3_363"], "1xT4", "ResNet-50 + Inception 363 (~29 min)"),
    "S3_1gpu_r50_r101_inc3": (["r50", "r101", "inc3"], "1xT4",
                              "single-GPU backup plan (in the plan trained 16 epochs; here 20-epoch probs)"),
}

pred_ref = {k: mean(v, test).argmax(1) for k, v in REF.items()}
rows = []
for name, (ms, hw, why) in CANDIDATES.items():
    pv, pt = mean(ms, val), mean(ms, test)
    c = pv.argmax(1) == y
    labels = pt.argmax(1)
    sub = pd.DataFrame({"id": test_table.id, "label": labels})
    assert sub.id.tolist() == test_table.id.tolist() and sub.label.between(0, 19).all() and len(sub) == 8299
    sub.to_csv(OUT / f"{name}.csv", index=False)
    t, g0, g1 = best_split(ms)
    rows.append({
        "file": f"{name}.csv", "models": " + ".join(ms), "hardware": hw,
        "train_min_total_T4": round(sum(full_min[m] for m in ms), 1),
        "per_gpu_min": (f"GPU0 {'+'.join(g0)} {sum(full_min[m] for m in g0):.1f} | GPU1 {'+'.join(g1)} "
                        f"{sum(full_min[m] for m in g1):.1f}") if hw == "2xT4" else f"{sum(full_min[m] for m in ms):.1f}",
        "val_errors": int((~c).sum()), "val_err_clean": int((~c[~degraded]).sum()),
        "val_err_degraded": int((~c[degraded]).sum()),
        "diff_vs_A": int((labels != pred_ref["A"]).sum()), "diff_vs_7": int((labels != pred_ref["7"]).sum()),
        "diff_vs_r50r101": int((labels != pred_ref["r50+r101"]).sum()), "why": why,
    })
table = pd.DataFrame(rows)
table.to_csv(OUT / "candidates.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 80)
print(table.drop(columns="why").to_string(index=False))

# sanity: candidate A must equal the CSV that scored 0.96678
old = pd.read_csv(RUNS / "submission_ensemble_inc3_r101_r152_r50.csv")
same = (old.label.to_numpy() == pd.read_csv(OUT / "A_2gpu_r50_r101_r152_inc3.csv").label.to_numpy()).mean()
print(f"\nA vs runs/submission_ensemble_inc3_r101_r152_r50.csv: {same:.4%} identical labels")
print("known public:", PUBLIC, "| differing labels between the known ones: A vs 7:",
      int((pred_ref['A'] != pred_ref['7']).sum()), " A vs r50+r101:", int((pred_ref['A'] != pred_ref['r50+r101']).sum()))
