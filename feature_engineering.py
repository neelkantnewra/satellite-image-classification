# ============================================================
# fMoW — Part 3: Feature Engineering
# Intensity Histogram + LBP + Sobel/Canny + HOG
# ============================================================

import cv2
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
from skimage.feature import local_binary_pattern, hog

INPUT_DIR   = Path("fmow_preprocessed")
OUTPUT_DIR  = Path("fmow_features")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Feature extraction functions ──────────────────────────────

def intensity_histogram(img, bins=64):
    """Pixel intensity distribution — 64 bins normalized."""
    hist, _ = np.histogram(img.flatten(), bins=bins, range=(0, 256))
    return hist / hist.sum()  # normalize


def lbp_histogram(img, P=8, R=1, bins=64):
    """Local Binary Pattern texture descriptor."""
    lbp = local_binary_pattern(img, P=P, R=R, method="uniform")
    hist, _ = np.histogram(lbp.flatten(), bins=bins, range=(0, P + 2))
    return hist / hist.sum()


def edge_features(img):
    """Sobel + Canny edge features."""
    # Sobel magnitude
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # Summarize sobel as histogram (16 bins)
    sobel_hist, _ = np.histogram(sobel_mag.flatten(), bins=16, range=(0, 1024))
    sobel_hist = sobel_hist / sobel_hist.sum()

    # Canny — scalar stats: edge pixel ratio + mean magnitude on edges
    canny = cv2.Canny(img, threshold1=50, threshold2=150)
    edge_ratio      = np.sum(canny > 0) / canny.size
    edge_mean_mag   = sobel_mag[canny > 0].mean() if edge_ratio > 0 else 0.0
    canny_features  = np.array([edge_ratio, edge_mean_mag])

    return np.concatenate([sobel_hist, canny_features])  # 18 features


def hog_features(img):
    """Histogram of Oriented Gradients — mid-level shape/structure."""
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False
    )
    return features


def extract_all(img_path):
    """Load preprocessed image and extract all features."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read: {img_path}")

    hist    = intensity_histogram(img)   # 64
    lbp     = lbp_histogram(img)         # 64
    edges   = edge_features(img)         # 18
    hog_f   = hog_features(img)          # 324 (for 128x128, 16x16 cells, 2x2 blocks)

    return np.concatenate([hist, lbp, edges, hog_f])


# ── Process all splits ────────────────────────────────────────
def process_split(split, label, class_id):
    src    = INPUT_DIR / split / label
    files  = sorted(src.glob("*.jpg"))
    rows   = []
    failed = 0

    for f in tqdm(files, desc=f"{split}/{label}"):
        try:
            features = extract_all(f)
            rows.append({
                "filename" : f.name,
                "label"    : label,
                "class_id" : class_id,  # 0=urban, 1=natural
                "split"    : split,
                **{f"f_{i}": v for i, v in enumerate(features)}
            })
        except Exception as e:
            print(f"  ⚠ Skipped {f.name}: {e}")
            failed += 1

    print(f"  ✅ {len(rows)} extracted, {failed} failed")
    return rows

all_rows = []

for split in ["train", "test"]:
    for label, cid in [("urban", 0), ("natural", 1)]:
        print(f"\nExtracting {split}/{label}...")
        all_rows.extend(process_split(split, label, cid))

# ── External image ────────────────────────────────────────────
print("\nExtracting external image...")
ext_files = sorted((INPUT_DIR / "external").glob("*.jpg"))
ext_rows  = []
for f in ext_files:
    features = extract_all(f)
    ext_rows.append({
        "filename" : f.name,
        "label"    : "urban",
        "class_id" : 0,
        "split"    : "external",
        **{f"f_{i}": v for i, v in enumerate(features)}
    })
print(f"  ✅ External done")

# ── Save ──────────────────────────────────────────────────────
df     = pd.DataFrame(all_rows)
df_ext = pd.DataFrame(ext_rows)

# CSV
df.to_csv(OUTPUT_DIR / "features.csv", index=False)
df_ext.to_csv(OUTPUT_DIR / "features_external.csv", index=False)

# NumPy arrays (for faster ML loading)
meta_cols    = ["filename", "label", "class_id", "split"]
feature_cols = [c for c in df.columns if c.startswith("f_")]

X_train = df[df.split == "train"][feature_cols].values
y_train = df[df.split == "train"]["class_id"].values
X_test  = df[df.split == "test"][feature_cols].values
y_test  = df[df.split == "test"]["class_id"].values
X_ext   = df_ext[feature_cols].values

np.save(OUTPUT_DIR / "X_train.npy", X_train)
np.save(OUTPUT_DIR / "y_train.npy", y_train)
np.save(OUTPUT_DIR / "X_test.npy",  X_test)
np.save(OUTPUT_DIR / "y_test.npy",  y_test)
np.save(OUTPUT_DIR / "X_external.npy", X_ext)

# Pickle (everything in one file)
with open(OUTPUT_DIR / "features.pkl", "wb") as f:
    pickle.dump({
        "X_train"      : X_train,
        "y_train"      : y_train,
        "X_test"       : X_test,
        "y_test"       : y_test,
        "X_external"   : X_ext,
        "feature_cols" : feature_cols,
    }, f)

# ── Summary ───────────────────────────────────────────────────
n_features = len(feature_cols)
print(f"""
✅ Part 3 Complete!

Feature vector breakdown:
  Intensity Histogram : 64  features
  LBP Texture         : 64  features
  Sobel + Canny Edges : 18  features
  HOG Descriptors     : {n_features - 146}  features
  ─────────────────────────────
  Total               : {n_features} features per image

Saved to fmow_features/
  ├── features.csv          ({len(df)} rows)
  ├── features_external.csv (1 row)
  ├── X_train.npy           {X_train.shape}
  ├── y_train.npy           {y_train.shape}
  ├── X_test.npy            {X_test.shape}
  ├── y_test.npy            {y_test.shape}
  ├── X_external.npy        {X_ext.shape}
  └── features.pkl          (all in one)
""")