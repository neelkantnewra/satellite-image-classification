# ============================================================
# fMoW — Part 2: Preprocessing
# Resize → Grayscale → CLAHE → Gaussian/Median Filter
# ============================================================

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

INPUT_DIR  = Path("fmow_data/splits")
OUTPUT_DIR = Path("fmow_preprocessed")
IMG_SIZE   = (128, 128)

# ── Create output folders ─────────────────────────────────────
for split in ["train", "test"]:
    for label in ["urban", "natural"]:
        (OUTPUT_DIR / split / label).mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "external").mkdir(parents=True, exist_ok=True)

# ── Preprocessing pipeline ────────────────────────────────────
def preprocess(img_path):
    # 1. Load
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read: {img_path}")

    # 2. Resize to 128x128
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # 3. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. CLAHE (better than basic histogram equalization — handles local contrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # 5. Gaussian filter (denoise while preserving edges)
    smoothed = cv2.GaussianBlur(equalized, (3, 3), sigmaX=0.5)

    return smoothed

# ── Process train & test splits ───────────────────────────────
for split in ["train", "test"]:
    for label in ["urban", "natural"]:
        src = INPUT_DIR / split / label
        dst = OUTPUT_DIR / split / label
        files = sorted(src.glob("*.jpg"))

        print(f"Processing {split}/{label} — {len(files)} images...")
        failed = 0
        for f in tqdm(files, desc=f"{split}/{label}"):
            try:
                processed = preprocess(f)
                cv2.imwrite(str(dst / f.name), processed)
            except Exception as e:
                print(f"  ⚠ Skipped {f.name}: {e}")
                failed += 1

        done = len(list(dst.glob("*.jpg")))
        print(f"  ✅ {done} saved, {failed} failed\n")

# ── Process external image ────────────────────────────────────
ext_files = sorted(Path("fmow_data/external").glob("*.jpg"))
print(f"Processing external image...")
for f in ext_files:
    processed = preprocess(f)
    cv2.imwrite(str(OUTPUT_DIR / "external" / f.name), processed)
print(f"  ✅ External image saved")

# ── Verify counts ─────────────────────────────────────────────
print(f"""
✅ Part 2 Complete!

fmow_preprocessed/
├── train/urban/     ({len(list((OUTPUT_DIR/'train/urban').glob('*.jpg')))} images)
├── train/natural/   ({len(list((OUTPUT_DIR/'train/natural').glob('*.jpg')))} images)
├── test/urban/      ({len(list((OUTPUT_DIR/'test/urban').glob('*.jpg')))} images)
├── test/natural/    ({len(list((OUTPUT_DIR/'test/natural').glob('*.jpg')))} images)
└── external/        ({len(list((OUTPUT_DIR/'external').glob('*.jpg')))} image)

All images: 128×128 grayscale, CLAHE enhanced, Gaussian smoothed
""")

# ── Quick visual sanity check (optional) ──────────────────────
import matplotlib.pyplot as plt

sample_urban   = sorted((OUTPUT_DIR / "train/urban").glob("*.jpg"))[0]
sample_natural = sorted((OUTPUT_DIR / "train/natural").glob("*.jpg"))[0]

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(cv2.imread(str(sample_urban),   0), cmap="gray")
axes[0].set_title("Urban — preprocessed")
axes[0].axis("off")
axes[1].imshow(cv2.imread(str(sample_natural), 0), cmap="gray")
axes[1].set_title("Natural — preprocessed")
axes[1].axis("off")
plt.tight_layout()
plt.savefig("preprocessing_sample.png", dpi=150)
plt.show()
print("Sample saved → preprocessing_sample.png")