# ============================================================
# fMoW — Part 4: Feature Selection
# Variance Thresholding + PCA
# ============================================================

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

FEATURE_DIR = Path("fmow_features")
OUTPUT_DIR  = Path("fmow_selected")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Load features ─────────────────────────────────────────────
print("Loading features...")
with open(FEATURE_DIR / "features.pkl", "rb") as f:
    data = pickle.load(f)

X_train    = data["X_train"]
y_train    = data["y_train"]
X_test     = data["X_test"]
y_test     = data["y_test"]
X_external = data["X_external"]

print(f"  X_train: {X_train.shape}")
print(f"  X_test:  {X_test.shape}")

# ── Step 1: Standard Scaling ──────────────────────────────────
# Must scale BEFORE variance threshold and PCA
print("\nScaling features...")
scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
X_test_scaled  = scaler.transform(X_test)
X_ext_scaled   = scaler.transform(X_external)

# ── Step 2: Variance Thresholding ────────────────────────────
print("\nApplying Variance Threshold...")
selector = VarianceThreshold(threshold=0.01)
X_train_vt = selector.fit_transform(X_train_scaled)
X_test_vt  = selector.transform(X_test_scaled)
X_ext_vt   = selector.transform(X_ext_scaled)

removed = X_train.shape[1] - X_train_vt.shape[1]
print(f"  Features before : {X_train.shape[1]}")
print(f"  Features removed: {removed} (low variance)")
print(f"  Features after  : {X_train_vt.shape[1]}")

# ── Step 3: PCA ───────────────────────────────────────────────
print("\nFitting PCA...")
pca = PCA(n_components=0.95, svd_solver="full")   # keep 95% variance
X_train_pca = pca.fit_transform(X_train_vt)
X_test_pca  = pca.transform(X_test_vt)
X_ext_pca   = pca.transform(X_ext_vt)

print(f"  Components kept : {pca.n_components_} (explains 95% variance)")
print(f"  X_train PCA     : {X_train_pca.shape}")
print(f"  X_test  PCA     : {X_test_pca.shape}")

# ── Step 4: Plot explained variance ───────────────────────────
cumvar = np.cumsum(pca.explained_variance_ratio_) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(cumvar, color="steelblue", linewidth=2)
axes[0].axhline(95, color="red", linestyle="--", label="95% threshold")
axes[0].axvline(pca.n_components_ - 1, color="green", linestyle="--",
                label=f"{pca.n_components_} components")
axes[0].set_xlabel("Number of Components")
axes[0].set_ylabel("Cumulative Explained Variance (%)")
axes[0].set_title("PCA — Cumulative Explained Variance")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(range(min(30, pca.n_components_)),
            pca.explained_variance_ratio_[:30] * 100,
            color="steelblue", alpha=0.8)
axes[1].set_xlabel("Principal Component")
axes[1].set_ylabel("Explained Variance (%)")
axes[1].set_title("Top 30 Components — Individual Variance")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_variance.png", dpi=150)
plt.show()
print("  Plot saved → fmow_selected/pca_variance.png")

# ── Step 5: Save everything ───────────────────────────────────
print("\nSaving...")
with open(OUTPUT_DIR / "selected_features.pkl", "wb") as f:
    pickle.dump({
        "X_train"    : X_train_pca,
        "y_train"    : y_train,
        "X_test"     : X_test_pca,
        "y_test"     : y_test,
        "X_external" : X_ext_pca,
        "scaler"     : scaler,
        "selector"   : selector,
        "pca"        : pca,
    }, f)

np.save(OUTPUT_DIR / "X_train_pca.npy", X_train_pca)
np.save(OUTPUT_DIR / "X_test_pca.npy",  X_test_pca)
np.save(OUTPUT_DIR / "y_train.npy",     y_train)
np.save(OUTPUT_DIR / "y_test.npy",      y_test)
np.save(OUTPUT_DIR / "X_external_pca.npy", X_ext_pca)

# ── Summary ───────────────────────────────────────────────────
print(f"""
✅ Part 4 Complete!

Pipeline:
  Raw features       : {X_train.shape[1]}
  After Var.Threshold: {X_train_vt.shape[1]}  ({removed} removed)
  After PCA (95% var): {pca.n_components_}  components

Saved to fmow_selected/
  ├── selected_features.pkl  (all arrays + fitted transformers)
  ├── X_train_pca.npy        {X_train_pca.shape}
  ├── X_test_pca.npy         {X_test_pca.shape}
  ├── y_train.npy            {y_train.shape}
  ├── y_test.npy             {y_test.shape}
  ├── X_external_pca.npy     {X_ext_pca.shape}
  └── pca_variance.png
""")