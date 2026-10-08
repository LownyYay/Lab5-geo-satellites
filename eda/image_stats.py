"""Per-image low-level statistics for train and val (test is NOT touched: rules allow test only for predictions).

Output: eda/out/image_stats.csv and eda/out/radial_spectrum.npz
"""
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "eda" / "out"
OUT.mkdir(parents=True, exist_ok=True)

LAPLACE = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
# Immerkaer noise estimator kernel (difference of two Laplacians, cancels smooth structure)
NOISE_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)

N = 256
yy, xx = np.indices((N, N))
RADIUS = np.hypot(yy - N // 2, xx - N // 2).astype(int)
N_BINS = N // 2
WINDOW = np.outer(np.hanning(N), np.hanning(N))


def jpeg_luma_qsum(path):
    data = path.read_bytes()
    i = 2
    while i < len(data) - 4 and data[i] == 0xFF:
        marker = data[i + 1]
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker == 0xDB:
            seg = data[i + 4:i + 2 + length]
            return int(sum(seg[1:65]))
        i += 2 + length
    return -1


def blockiness(gray):
    """Mean abs difference across 8x8 JPEG block borders divided by the same inside blocks (>1 => visible blocks)."""
    dx = np.abs(np.diff(gray, axis=1))
    border = dx[:, 7::8].mean()
    inside = np.delete(dx, np.s_[7::8], axis=1).mean()
    return border / (inside + 1e-9)


def stats_for(path):
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    gray = img @ np.array([0.299, 0.587, 0.114])
    hsv = np.asarray(Image.open(path).convert("HSV"), dtype=np.float64)

    lap = ndimage.convolve(gray, LAPLACE, mode="reflect")
    noise = ndimage.convolve(gray, NOISE_K, mode="reflect")[1:-1, 1:-1]
    sigma_noise = np.sqrt(np.pi / 2) * np.abs(noise).mean() / 6.0

    spec = np.abs(np.fft.fftshift(np.fft.fft2((gray - gray.mean()) * WINDOW))) ** 2
    radial = np.bincount(RADIUS.ravel(), spec.ravel(), minlength=N)[:N_BINS] / np.bincount(RADIUS.ravel(), minlength=N)[:N_BINS]
    total = radial[1:].sum()
    hf_ratio = radial[N_BINS // 2:].sum() / total          # share of energy above half Nyquist
    mf_ratio = radial[N_BINS // 4:N_BINS // 2].sum() / total

    row = {
        "w": img.shape[1], "h": img.shape[0],
        "mean_r": img[..., 0].mean(), "mean_g": img[..., 1].mean(), "mean_b": img[..., 2].mean(),
        "std_gray": gray.std(), "sat_mean": hsv[..., 1].mean(),
        "p01": np.percentile(gray, 1), "p99": np.percentile(gray, 99),
        "lap_var": lap.var(), "noise_sigma": sigma_noise,
        "hf_ratio": hf_ratio, "mf_ratio": mf_ratio,
        "blockiness": blockiness(gray),
        "file_kb": path.stat().st_size / 1024, "q_luma_sum": jpeg_luma_qsum(path),
    }
    return row, np.log10(radial + 1e-9)


def main():
    classes = pd.read_csv(DATA / "classes.csv").sort_values("label").class_name.tolist()
    rows, spectra = [], {"train": [], "val": []}
    for split in ("train", "val"):
        table = pd.read_csv(DATA / f"{split}.csv", dtype={"id": str})
        for image_id, label in zip(table.id, table.label):
            row, spec = stats_for(DATA / split / image_id)
            row.update(split=split, id=image_id, label=label, class_name=classes[label])
            rows.append(row)
            spectra[split].append(spec)
        print(split, "done", len(table))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "image_stats.csv", index=False)
    labels = {s: df[df.split == s].label.to_numpy() for s in spectra}
    np.savez(OUT / "radial_spectrum.npz", train=np.array(spectra["train"]), val=np.array(spectra["val"]),
             train_labels=labels["train"], val_labels=labels["val"])
    print("saved", OUT)


if __name__ == "__main__":
    main()
