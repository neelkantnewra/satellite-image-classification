# ============================================================
# fMoW — Part 5: Model Training
# SVM + Random Forest with 5-Fold Cross Validation
# ============================================================

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

OUTPUT_DIR = Path("fmow_models")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Load selected features ────────────────────────────────────
print("Loading features...")
with open("fmow_selected/selected_features.pkl", "rb") as f:
    data = pickle.load(f)

X_train    = data["X_train"]
y_train    = data["y_train"]
X_test     = data["X_test"]
y_test     = data["y_test"]
X_external = data["X_external"]

print(f"  X_train : {X_train.shape}")
print(f"  X_test  : {X_test.shape}")

# ── Scoring metrics ───────────────────────────────────────────
scoring = {
    "accuracy"  : make_scorer(accuracy_score),
    "precision" : make_scorer(precision_score, zero_division=0),
    "recall"    : make_scorer(recall_score,    zero_division=0),
    "f1"        : make_scorer(f1_score,        zero_division=0),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Model 1: SVM ──────────────────────────────────────────────
print("\nTraining SVM (5-fold CV with Grid Search)...")
from sklearn.model_selection import GridSearchCV

param_grid = {
    "C"     : [1, 10, 50, 100],
    "gamma" : ["scale", "auto", 0.01, 0.001]
}
grid = GridSearchCV(SVC(probability=True, random_state=42),
                    param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)
print(f"Best SVM params: {grid.best_params_}")
svm = grid.best_estimator_
svm_cv = cross_validate(svm, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)

print(f"  Accuracy  : {svm_cv['test_accuracy'].mean():.4f} ± {svm_cv['test_accuracy'].std():.4f}")
print(f"  Precision : {svm_cv['test_precision'].mean():.4f} ± {svm_cv['test_precision'].std():.4f}")
print(f"  Recall    : {svm_cv['test_recall'].mean():.4f} ± {svm_cv['test_recall'].std():.4f}")
print(f"  F1        : {svm_cv['test_f1'].mean():.4f} ± {svm_cv['test_f1'].std():.4f}")

# Fit final SVM on full training set
print("  Fitting final SVM on full train set...")
svm.fit(X_train, y_train)

# ── Model 2: Random Forest ────────────────────────────────────
print("\nTraining Random Forest (5-fold CV)...")
rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                             min_samples_split=2, random_state=42, n_jobs=-1)
rf_cv = cross_validate(rf, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)

print(f"  Accuracy  : {rf_cv['test_accuracy'].mean():.4f} ± {rf_cv['test_accuracy'].std():.4f}")
print(f"  Precision : {rf_cv['test_precision'].mean():.4f} ± {rf_cv['test_precision'].std():.4f}")
print(f"  Recall    : {rf_cv['test_recall'].mean():.4f} ± {rf_cv['test_recall'].std():.4f}")
print(f"  F1        : {rf_cv['test_f1'].mean():.4f} ± {rf_cv['test_f1'].std():.4f}")

# Fit final RF on full training set
print("  Fitting final RF on full train set...")
rf.fit(X_train, y_train)

# ── Plot CV results ───────────────────────────────────────────
metrics = ["accuracy", "precision", "recall", "f1"]
svm_means = [svm_cv[f"test_{m}"].mean() for m in metrics]
svm_stds  = [svm_cv[f"test_{m}"].std()  for m in metrics]
rf_means  = [rf_cv[f"test_{m}"].mean()  for m in metrics]
rf_stds   = [rf_cv[f"test_{m}"].std()   for m in metrics]

x     = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, svm_means, width, yerr=svm_stds,
               label="SVM", color="steelblue", capsize=5, alpha=0.85)
bars2 = ax.bar(x + width/2, rf_means,  width, yerr=rf_stds,
               label="Random Forest", color="darkorange", capsize=5, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([m.capitalize() for m in metrics])
ax.set_ylabel("Score")
ax.set_title("5-Fold Cross Validation — SVM vs Random Forest")
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

# Add value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cv_comparison.png", dpi=150)
plt.show()
print("Plot saved → fmow_models/cv_comparison.png")

# ── Save models ───────────────────────────────────────────────
with open(OUTPUT_DIR / "models.pkl", "wb") as f:
    pickle.dump({
        "svm"        : svm,
        "rf"         : rf,
        "svm_cv"     : svm_cv,
        "rf_cv"      : rf_cv,
        "X_test"     : X_test,
        "y_test"     : y_test,
        "X_external" : X_external,
    }, f)

# ── Summary ───────────────────────────────────────────────────
print(f"""
✅ Part 5 Complete!

                  SVM              Random Forest
  Accuracy  :  {svm_cv['test_accuracy'].mean():.4f} ± {svm_cv['test_accuracy'].std():.4f}    {rf_cv['test_accuracy'].mean():.4f} ± {rf_cv['test_accuracy'].std():.4f}
  Precision :  {svm_cv['test_precision'].mean():.4f} ± {svm_cv['test_precision'].std():.4f}    {rf_cv['test_precision'].mean():.4f} ± {rf_cv['test_precision'].std():.4f}
  Recall    :  {svm_cv['test_recall'].mean():.4f} ± {svm_cv['test_recall'].std():.4f}    {rf_cv['test_recall'].mean():.4f} ± {rf_cv['test_recall'].std():.4f}
  F1        :  {svm_cv['test_f1'].mean():.4f} ± {svm_cv['test_f1'].std():.4f}    {rf_cv['test_f1'].mean():.4f} ± {rf_cv['test_f1'].std():.4f}

Saved to fmow_models/
  ├── models.pkl
  └── cv_comparison.png
""")