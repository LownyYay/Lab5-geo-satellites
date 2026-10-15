"""Critical analysis of the saved team artifacts (numpy / pandas / scipy only, no torch).

Uses ONLY: train/val labels, train/val image statistics (eda/out/image_stats.csv has no test rows),
runs/<model>/{val_probs.npy, history_*.csv, summary.json}. From test_probs.npy we only count how many
test labels differ between models / candidate ensembles (no statistics of test images or test predictions
are used to choose anything).

usage: python analyze.py            -> prints everything and writes ../out/*.csv
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

LAB = Path(__file__).resolve().parents[2]
RUNS, DATA = LAB / "runs", LAB / "data"
OUT = Path(__file__).resolve().parents[1] / "out"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)

val_table = pd.read_csv(DATA / "val.csv", dtype={"id": str})
y = val_table.label.to_numpy()
classes = pd.read_csv(DATA / "classes.csv").sort_values("label").class_name.tolist()
degraded = np.load(RUNS / "val_degraded.npy")
N = len(y)

MODELS = ["r50", "r101", "r152", "inc3", "r50_320", "r101_320", "inc3_363", "r50_mix", "inc3_363_mix", "r152_mix"]
summ = {m: json.loads((RUNS / m / "summary.json").read_text(encoding="utf-8")) for m in MODELS
        if (RUNS / m / "summary.json").exists()}
val = {m: np.load(RUNS / m / "val_probs.npy") for m in MODELS}
test = {m: np.load(RUNS / m / "test_probs.npy") for m in MODELS if (RUNS / m / "test_probs.npy").exists()}
# full-training minutes on Colab T4 (r152_mix has no summary: take the sum of its history)
full_min = {m: summ[m]["full_train_min"] if m in summ else pd.read_csv(RUNS / m / "history_full.csv").minutes.sum()
            for m in MODELS}
pred_min = {m: summ[m].get("predict_min", np.nan) if m in summ else np.nan for m in MODELS}


def nll(p):
    return float(-np.log(np.clip(p[np.arange(N), y], 1e-12, 1)).mean())


def ece(p, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


# ------------------------------------------------------------------------------------------------ 1. models
print("=" * 100, "\n1. single models (holdout -> val, chosen TTA variant)\n")
rows = []
for m in MODELS:
    p = val[m]
    c = p.argmax(1) == y
    rows.append({"model": m, "val_acc": c.mean(), "errors": int((~c).sum()), "err_clean": int((~c[~degraded]).sum()),
                 "err_degraded": int((~c[degraded]).sum()), "acc_clean": c[~degraded].mean(),
                 "acc_degr": c[degraded].mean(), "nll": nll(p), "ece": ece(p), "mean_conf": p.max(1).mean(),
                 "full_train_min": full_min[m], "predict_min": pred_min[m], "has_test": m in test})
single = pd.DataFrame(rows).set_index("model")
print(single.round(4).to_string())
single.to_csv(OUT / "single_models.csv")

# ------------------------------------------------------------------------------------------------ 2. ensembles A / 7
A = ["r50", "r101", "r152", "inc3"]
SEVEN = A + ["r50_320", "r101_320", "inc3_363"]


def ens(ms, d=val):
    return np.mean([d[m] for m in ms], axis=0)


pA = ens(A)
cA = pA.argmax(1) == y
print("\n4-model ensemble A:", (~cA).sum(), "errors; clean", (~cA[~degraded]).sum(), "/", (~degraded).sum(),
      "degraded", (~cA[degraded]).sum(), "/", degraded.sum(), f"NLL {nll(pA):.4f} ECE {ece(pA):.3f}")
p7 = ens(SEVEN)
print("7-model ensemble   :", (p7.argmax(1) != y).sum(), "errors", f"NLL {nll(p7):.4f} ECE {ece(p7):.3f}")

# oracle / majority structure of errors over the 9 models with test probs
pool = [m for m in MODELS if m in test]
wrong = np.stack([val[m].argmax(1) != y for m in pool])          # (models, N)
n_wrong = wrong.sum(0)
print("\nhow many of the", len(pool), "models are wrong on each val image:")
print(pd.Series(n_wrong).value_counts().sort_index().to_string())
print("images wrong for ALL models:", (n_wrong == len(pool)).sum(), "| wrong for >= half:",
      (n_wrong >= len(pool) / 2).sum(), "| wrong for at least one:", (n_wrong > 0).sum())

# ------------------------------------------------------------------------------------------------ 3. errors of A
print("\n" + "=" * 100, "\n3. errors of ensemble A by class (true -> predicted)\n")
pred = pA.argmax(1)
err = pd.DataFrame({"true": np.array(classes)[y[~cA]], "pred": np.array(classes)[pred[~cA]],
                    "degraded": degraded[~cA], "conf": pA.max(1)[~cA], "p_true": pA[np.arange(N), y][~cA]})
print(err.sort_values(["true", "pred"]).round(3).to_string(index=False))
per_class = pd.DataFrame({
    "class": classes,
    "n_val": np.bincount(y, minlength=20),
    "err_A": np.bincount(y[~cA], minlength=20),
    "err_sum_9models": wrong.sum(0) @ np.eye(20)[y] if False else np.array([wrong[:, y == k].sum() for k in range(20)]),
})
per_class["err_rate_9models_%"] = 100 * per_class.err_sum_9models / (per_class.n_val * len(pool))
print("\nper-class error, summed over the 9 single models (each model counts):")
print(per_class.sort_values("err_sum_9models", ascending=False).round(2).to_string(index=False))
per_class.to_csv(OUT / "per_class.csv", index=False)

# ------------------------------------------------------------------------------------------------ 4. degradation types
print("\n" + "=" * 100, "\n4. degradation types (flags recomputed from train statistics only)\n")
stats = pd.read_csv(LAB / "eda" / "out" / "image_stats.csv")
assert set(stats.split) == {"train", "val"}
tr = stats[stats.split == "train"]
q = tr.groupby("label").agg(lap_q01=("lap_var", lambda s: s.quantile(0.01)),
                            lap_q99=("lap_var", lambda s: s.quantile(0.99)),
                            blk_q99=("blockiness", lambda s: s.quantile(0.99)),
                            std_q01=("std_gray", lambda s: s.quantile(0.01)),
                            mf_q02=("mf_ratio", lambda s: s.quantile(0.02)))
v = stats[stats.split == "val"].set_index("id").loc[val_table.id].join(q, on="label")
types = pd.DataFrame({
    "blur/low-res": (v.lap_var < v.lap_q01).to_numpy() | (v.mf_ratio < v.mf_q02).to_numpy(),
    "noise/oversharp": (v.lap_var > v.lap_q99).to_numpy(),
    "jpeg-blocky": (v.blockiness > v.blk_q99).to_numpy(),
    "gray": (v.sat_mean < 8).to_numpy(),
    "low-contrast": (v.std_gray < v.std_q01).to_numpy(),
})
any_type = types.any(axis=1).to_numpy()
print(f"recomputed 'any type' flag: {any_type.sum()} images; saved mask: {degraded.sum()}; "
      f"agreement {np.mean(any_type == degraded):.3f}")
rows = []
for t in list(types.columns) + ["ANY (saved mask)", "clean (saved mask)"]:
    msk = degraded if t == "ANY (saved mask)" else (~degraded if t.startswith("clean") else types[t].to_numpy())
    row = {"type": t, "n": int(msk.sum()), "err_A": int((~cA[msk]).sum()), "acc_A": cA[msk].mean()}
    row["mean_single_acc"] = np.mean([(val[m].argmax(1) == y)[msk].mean() for m in pool])
    rows.append(row)
deg = pd.DataFrame(rows)
print(deg.round(4).to_string(index=False))
deg.to_csv(OUT / "degradation_types.csv", index=False)

# counterfactual: how much would the test score drop if test had MORE degraded images at val per-group accuracy?
acc_c, acc_d = cA[~degraded].mean(), cA[degraded].mean()
print("\ncounterfactual for ensemble A (val per-group accuracy, varying the degraded share):")
for share in (0.272, 0.35, 0.45, 0.60):
    print(f"  degraded share {share:.0%}: expected acc {share * acc_d + (1 - share) * acc_c:.4f}")
# and with 2x the error rate on degraded images
print(f"  27% degraded but degraded error x2: {0.272 * (1 - 2 * (1 - acc_d)) + 0.728 * acc_c:.4f}")

# ------------------------------------------------------------------------------------------------ 5. confidence
print("\n" + "=" * 100, "\n5. confidence of ensemble A on val\n")
conf = pA.max(1)
srt = np.sort(pA, 1)
margin = srt[:, -1] - srt[:, -2]
bins = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
cb = pd.cut(conf, bins)
print(pd.DataFrame({"conf_bin": cb, "correct": cA}).groupby("conf_bin", observed=True).correct.agg(["size", "mean"])
      .rename(columns={"size": "n", "mean": "acc"}).round(3).to_string())
print(f"errors with margin < 0.2: {(~cA & (margin < 0.2)).sum()} of {(~cA).sum()}; "
      f"correct with margin < 0.2: {(cA & (margin < 0.2)).sum()}")
print("true class is top-2 for", (np.argsort(-pA, 1)[:, :2] == y[:, None]).any(1)[~cA].sum(), "of", (~cA).sum(), "errors")

# ------------------------------------------------------------------------------------------------ 6. diversity
print("\n" + "=" * 100, "\n6. pairwise diversity (val: both wrong / only one wrong / Yule Q; test: labels that differ)\n")
rows = []
for a, b in itertools.combinations(pool, 2):
    wa, wb = val[a].argmax(1) != y, val[b].argmax(1) != y
    n11, n00 = int((wa & wb).sum()), int((~wa & ~wb).sum())
    n10, n01 = int((wa & ~wb).sum()), int((~wa & wb).sum())
    Q = (n00 * n11 - n01 * n10) / max(1, n00 * n11 + n01 * n10)
    rows.append({"a": a, "b": b, "both_wrong": n11, "only_a_wrong": n10, "only_b_wrong": n01,
                 "val_disagree": int((val[a].argmax(1) != val[b].argmax(1)).sum()), "yule_Q": Q,
                 "test_disagree": int((test[a].argmax(1) != test[b].argmax(1)).sum())})
pair = pd.DataFrame(rows)
print(pair.round(3).to_string(index=False))
pair.to_csv(OUT / "pairwise.csv", index=False)
fam = {"r50": "res", "r101": "res", "r152": "res", "r50_320": "res", "r101_320": "res", "r50_mix": "res",
       "inc3": "inc", "inc3_363": "inc", "inc3_363_mix": "inc"}
pair["kind"] = [("res-res" if fam[a] == fam[b] == "res" else "inc-inc" if fam[a] == fam[b] == "inc" else "res-inc")
                for a, b in zip(pair.a, pair.b)]
print("\nmean by family pair:")
print(pair.groupby("kind")[["both_wrong", "val_disagree", "yule_Q", "test_disagree"]].mean().round(2).to_string())

# ------------------------------------------------------------------------------------------------ 7. subsets
print("\n" + "=" * 100, "\n7. all ensembles of the 9 models with test probs (equal-weight mean of probabilities)\n")
predA_test = ens(A, test).argmax(1)


def makespan(costs):
    """best split of the models over 2 GPUs: minimal max(sum on GPU0, sum on GPU1)"""
    best = sum(costs)
    for r in range(len(costs) + 1):
        for part in itertools.combinations(range(len(costs)), r):
            s0 = sum(costs[i] for i in part)
            best = min(best, max(s0, sum(costs) - s0))
    return best


rows = []
for r in range(1, len(pool) + 1):
    for ms in itertools.combinations(pool, r):
        p = ens(ms)
        c = p.argmax(1) == y
        costs = [full_min[m] for m in ms]
        rows.append({"models": "+".join(ms), "k": r, "val_err": int((~c).sum()), "err_degr": int((~c[degraded]).sum()),
                     "nll": nll(p), "gpu_min_sum": sum(costs), "makespan_2gpu": makespan(costs),
                     "diff_vs_A_test": int((ens(ms, test).argmax(1) != predA_test).sum())})
sub = pd.DataFrame(rows)
sub.to_csv(OUT / "subsets.csv", index=False)
print("val errors over all", len(sub), "subsets with >= 3 models: ",
      sub[sub.k >= 3].val_err.describe().round(2).to_dict())
print("\nbest 15 subsets by val errors (ties -> lower NLL):")
print(sub.sort_values(["val_err", "nll"]).head(15).round(4).to_string(index=False))
for budget in (36, 40):
    print(f"\nfeasible on ONE T4 (sum of Colab full-train minutes <= {budget}), best 10 by val errors:")
    print(sub[sub.gpu_min_sum <= budget].sort_values(["val_err", "nll"]).head(10).round(4).to_string(index=False))
    print(f"\nfeasible on TWO T4 (makespan <= {budget}), best 10 by val errors:")
    print(sub[sub.makespan_2gpu <= budget].sort_values(["val_err", "nll"]).head(10).round(4).to_string(index=False))

# paired comparison: is any subset significantly better than A on val?
best_row = sub.sort_values(["val_err", "nll"]).iloc[0]
pb = ens(best_row.models.split("+"))
cb_ = pb.argmax(1) == y
b_, c_ = int((cA & ~cb_).sum()), int((~cA & cb_).sum())
print(f"\nMcNemar A vs best-by-val subset ({best_row.models}): A-only-right {b_}, best-only-right {c_}, "
      f"exact p = {st.binomtest(min(b_, c_), b_ + c_, 0.5).pvalue if b_ + c_ else 1:.3f}")

# ------------------------------------------------------------------------------------------------ 8. val -> LB gap
print("\n" + "=" * 100, "\n8. val -> public gap\n")
for name, errs, pub in [("ensemble A", int((~cA).sum()), 0.96678), ("inc3_363 alone", int(single.loc["inc3_363", "errors"]), 0.95989),
                        ("r50+r101", int((ens(["r50", "r101"]).argmax(1) != y).sum()), 0.96506)]:
    # probability of seeing <= errs val errors if the true error rate equals the public one
    p_low = st.binom.cdf(errs, N, 1 - pub)
    print(f"{name:15s} val errors {errs:2d} (acc {1 - errs / N:.4f}); public {pub:.5f}; "
          f"P(val errors <= {errs} | true acc = public) = {p_low:.3f}; expected val errors {N * (1 - pub):.1f}")
print("public SE (n=5810) at 0.967:", round(np.sqrt(0.967 * 0.033 / 5810) * 100, 3), "pp;",
      "private SE (n~2489):", round(np.sqrt(0.967 * 0.033 / 2489) * 100, 3), "pp =",
      round(np.sqrt(0.967 * 0.033 * 2489), 1), "images")

# ------------------------------------------------------------------------------------------------ 9. curves
print("\n" + "=" * 100, "\n9. learning curves (holdout, val without TTA)\n")
rows = []
for m in MODELS:
    h = pd.read_csv(RUNS / m / "history_holdout.csv")
    E = len(h)
    rec = {"model": m, "epochs": E}
    for e in (8, 10, 12, 14, 16, 18, E):
        if e <= E:
            rec[f"acc@{e}"] = h.val_acc.iloc[max(0, e - 2):e + 1].mean()   # 3-epoch mean around e (noise)
    rec["acc_last3"] = h.val_acc.iloc[-3:].mean()
    rec["loss_min_epoch"] = int(h.val_loss.idxmin()) + 1
    rec["loss_last"] = h.val_loss.iloc[-1]
    rec["loss_min"] = h.val_loss.min()
    rows.append(rec)
curves = pd.DataFrame(rows)
print(curves.round(4).to_string(index=False))
curves.to_csv(OUT / "curves_summary.csv", index=False)
plain = curves[~curves.model.str.contains("mix")]
print("\nmean over the 7 plain (20-epoch) models:")
print(plain[[c for c in plain.columns if c.startswith("acc@") or c == "acc_last3"]].mean().round(4).to_string())

# ------------------------------------------------------------------------------------------------ 10. timing
print("\n" + "=" * 100, "\n10. timing model (Colab T4, full training on 5698 images)\n")
GF = {"resnet50": 4.09, "resnet101": 7.80, "resnet152": 11.51, "inception_v3": 5.71}   # GFLOPs fwd at 224 (299 inc3)
BASE = {"resnet50": 224, "resnet101": 224, "resnet152": 224, "inception_v3": 299}
rows = []
for m in MODELS:
    s = summ.get(m)
    arch = s["arch"] if s else "resnet152"
    size = s["size"] if s else 256
    h = pd.read_csv(RUNS / m / "history_full.csv")
    rows.append({"model": m, "arch": arch, "size": size, "gflops": GF[arch] * (size / BASE[arch]) ** 2,
                 "min_per_epoch": h.minutes.mean(), "epochs": len(h), "full_min": full_min[m],
                 "predict_min": pred_min[m]})
tim = pd.DataFrame(rows)
res = tim[tim.arch.str.startswith("resnet")]
slope, intercept, r, *_ = st.linregress(res.gflops, res.min_per_epoch)
tim["fit_min_per_epoch"] = intercept + slope * tim.gflops
print(tim.round(3).to_string(index=False))
print(f"\nResNets: min/epoch = {intercept:.3f} + {slope:.4f} * GFLOPs   (r = {r:.4f})")
print("=> per epoch:", round(intercept * 60, 1), "s fixed +", round(slope * 60, 2), "s per GFLOP of the forward pass")
tim.to_csv(OUT / "timing.csv", index=False)
ps = tim.dropna(subset=["predict_min"])
ps = ps[ps.model.isin(["r50", "r101", "r152", "r50_320", "r101_320"])]
# predict: one test size, TTA x8 at the chosen size
ps = ps.assign(test_gf=[GF[a] * (summ[m]["test_sizes"][0] / BASE[a]) ** 2 for a, m in zip(ps.arch, ps.model)])
s2, i2, r2, *_ = st.linregress(ps.test_gf, ps.predict_min)
print(f"predict (TTA x8, 8299 imgs, incl. loading test): min = {i2:.2f} + {s2:.3f} * GFLOPs at test size (r = {r2:.3f})")
