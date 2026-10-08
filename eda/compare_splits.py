"""Compare train vs val image statistics within each class (controls for class composition).

For every statistic: per-class medians in train and val, the val/train ratio, a sign test over 20 classes,
and a pooled Mann-Whitney AUC (probability that a random val image has a larger value than a random train image).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "eda" / "out"

df = pd.read_csv(OUT / "image_stats.csv")
print("sizes:", df.groupby("split")[["w", "h"]].agg(["min", "max"]).to_string())
print("JPEG luma quant-table sums:", df.groupby("split").q_luma_sum.value_counts().to_string())
print()
print("class counts:\n", pd.crosstab(df.class_name, df.split).T.to_string())
print()

features = ["mean_r", "mean_g", "mean_b", "std_gray", "sat_mean", "p01", "p99", "lap_var", "noise_sigma",
            "hf_ratio", "mf_ratio", "blockiness", "file_kb"]
rows = []
for f in features:
    med = df.groupby(["class_name", "split"])[f].median().unstack()
    ratio = med["val"] / med["train"]
    n_up = int((med["val"] > med["train"]).sum())
    sign_p = stats.binomtest(n_up, len(med), 0.5).pvalue
    # within-class rank AUC averaged over classes
    aucs = []
    for c, g in df.groupby("class_name"):
        a, b = g[g.split == "val"][f], g[g.split == "train"][f]
        u = stats.mannwhitneyu(a, b).statistic
        aucs.append(u / (len(a) * len(b)))
    rows.append({"feature": f, "train_median": df[df.split == "train"][f].median(),
                 "val_median": df[df.split == "val"][f].median(), "median_class_ratio": ratio.median(),
                 "classes_val_higher": f"{n_up}/20", "sign_test_p": sign_p,
                 "within_class_AUC": np.mean(aucs)})
summary = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(summary.round(4).to_string(index=False))
summary.to_csv(OUT / "split_comparison.csv", index=False)

# per-class sharpness table
med = df.groupby(["class_name", "split"])[["lap_var", "hf_ratio", "noise_sigma", "sat_mean"]].median().unstack()
print()
print(med.round(3).to_string())

# radial spectrum, class-balanced average
spec = np.load(OUT / "radial_spectrum.npz")
def class_mean(s, y):
    return np.mean([s[y == c].mean(0) for c in np.unique(y)], axis=0)
tr, va = class_mean(spec["train"], spec["train_labels"]), class_mean(spec["val"], spec["val_labels"])
freq = np.arange(len(tr)) / (2 * len(tr))   # cycles per pixel, Nyquist = 0.5
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(freq[1:], tr[1:], label="train"); axes[0].plot(freq[1:], va[1:], label="val")
axes[0].set_xlabel("spatial frequency, cycles/pixel"); axes[0].set_ylabel("log10 power"); axes[0].legend()
axes[0].set_title("Radial power spectrum (class-balanced mean)")
axes[1].plot(freq[1:], va[1:] - tr[1:]); axes[1].axhline(0, color="gray", lw=0.8)
axes[1].set_xlabel("spatial frequency, cycles/pixel"); axes[1].set_ylabel("log10(val / train)")
axes[1].set_title("Val minus train")
plt.tight_layout(); plt.savefig(OUT / "radial_spectrum.png", dpi=110); plt.close()
np.savetxt(OUT / "radial_spectrum_diff.txt", np.c_[freq, tr, va, va - tr], fmt="%.4f",
           header="freq train val val-train")
print("\nlog10(val/train) power at freq 0.05/0.1/0.2/0.3/0.4/0.49:",
      [round(float(np.interp(q, freq, va - tr)), 3) for q in (0.05, 0.1, 0.2, 0.3, 0.4, 0.49)])

# visual grid: 4 train + 4 val images for several classes
classes = pd.read_csv(DATA / "classes.csv").sort_values("label").class_name.tolist()
show = list(range(20))
fig, axes = plt.subplots(len(show), 8, figsize=(16, 2 * len(show)))
for r, lab in enumerate(show):
    for k, split in enumerate(["train"] * 4 + ["val"] * 4):
        ids = df[(df.split == split) & (df.label == lab)].id.tolist()
        img = Image.open(DATA / split / ids[k % 4])
        axes[r, k].imshow(img); axes[r, k].axis("off")
        axes[r, k].set_title(f"{split}" if r == 0 else "", fontsize=9)
    axes[r, 0].text(-10, 128, classes[lab], ha="right", va="center", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "train_vs_val_grid.jpg", dpi=60); plt.close()
print("figures saved")
