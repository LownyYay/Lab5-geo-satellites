"""Is a SUBSET of val images degraded? For each val image, compute the percentile of its statistic inside the
TRAIN distribution of the same class. Under "same distribution" these percentiles are Uniform(0,1);
an excess near 0 or 1 reveals a corrupted subset. Train percentiles are computed leave-one-out as a reference.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "eda" / "out"
df = pd.read_csv(OUT / "image_stats.csv")

features = ["mf_ratio", "hf_ratio", "lap_var", "noise_sigma", "sat_mean", "std_gray", "mean_g", "blockiness", "p99", "p01"]
for f in features:
    pct = np.empty(len(df))
    for lab, g in df.groupby("label"):
        ref = np.sort(g[g.split == "train"][f].to_numpy())
        for idx, v, s in zip(g.index, g[f], g.split):
            r = ref if s == "val" else np.delete(ref, np.searchsorted(ref, v))   # leave-one-out for train
            pct[df.index.get_loc(idx)] = np.searchsorted(r, v, side="right") / len(r)
    df[f"pct_{f}"] = pct

rows = []
for f in features:
    for s in ("train", "val"):
        p = df[df.split == s][f"pct_{f}"]
        rows.append({"feature": f, "split": s, "<1%": (p < 0.01).mean(), "<5%": (p < 0.05).mean(),
                     ">95%": (p > 0.95).mean(), ">99%": (p > 0.99).mean()})
tails = pd.DataFrame(rows)
print("Share of images in the extreme tails of the TRAIN class distribution (uniform => 1%, 5%, 5%, 1%):")
print(tails.round(3).to_string(index=False))

print("\nGrayscale-like images (sat_mean < 8):")
print(df.assign(gray=df.sat_mean < 8).groupby("split").gray.agg(["sum", "mean"]).round(4).to_string())

# val images in the bottom 2% of their class for mid-frequency energy
blurry = df[(df.split == "val") & (df.pct_mf_ratio < 0.02)]
print(f"\nval images below the 2nd percentile of train mid-frequency energy: {len(blurry)} / 699 "
      f"({len(blurry) / 699:.1%}); expected under no shift: {0.02:.0%}")
print(blurry.class_name.value_counts().to_string())
df.to_csv(OUT / "image_stats_pct.csv", index=False)

# show val images with the most extreme statistics of each kind
picks = {
    "blurriest (mf_ratio pct)": df[df.split == "val"].nsmallest(8, "pct_mf_ratio"),
    "least saturated": df[df.split == "val"].nsmallest(8, "sat_mean"),
    "noisiest (noise_sigma pct)": df[df.split == "val"].nlargest(8, "pct_noise_sigma"),
    "lowest contrast (std pct)": df[df.split == "val"].nsmallest(8, "pct_std_gray"),
    "highest contrast (std pct)": df[df.split == "val"].nlargest(8, "pct_std_gray"),
    "most blocky": df[df.split == "val"].nlargest(8, "blockiness"),
}
fig, axes = plt.subplots(len(picks), 8, figsize=(16, 2.2 * len(picks)))
for r, (title, sub) in enumerate(picks.items()):
    for k, (_, row) in enumerate(sub.iterrows()):
        axes[r, k].imshow(Image.open(DATA / "val" / row.id)); axes[r, k].axis("off")
        axes[r, k].set_title(row.class_name[:14], fontsize=8)
    axes[r, 0].text(-20, 128, title, ha="right", va="center", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "val_extremes.jpg", dpi=60); plt.close()

# same for train, as a reference (train is not supposed to contain corrupted images)
picks_tr = {
    "blurriest (mf_ratio pct)": df[df.split == "train"].nsmallest(8, "pct_mf_ratio"),
    "least saturated": df[df.split == "train"].nsmallest(8, "sat_mean"),
    "noisiest (noise_sigma pct)": df[df.split == "train"].nlargest(8, "pct_noise_sigma"),
}
fig, axes = plt.subplots(len(picks_tr), 8, figsize=(16, 2.2 * len(picks_tr)))
for r, (title, sub) in enumerate(picks_tr.items()):
    for k, (_, row) in enumerate(sub.iterrows()):
        axes[r, k].imshow(Image.open(DATA / "train" / row.id)); axes[r, k].axis("off")
        axes[r, k].set_title(row.class_name[:14], fontsize=8)
    axes[r, 0].text(-20, 128, title, ha="right", va="center", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "train_extremes.jpg", dpi=60); plt.close()
print("figures saved")
