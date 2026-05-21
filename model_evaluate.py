# ============================================================
# fMoW — Part 6: Model Evaluation
# Test Set + Confusion Matrix + External Image Prediction
# ============================================================

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

OUTPUT_DIR = Path("fmow_evaluation")
OUTPUT_DIR.mkdir(exist_ok=True)

LABELS    = ["Urban", "Natural"]
CLASS_IDS = [0, 1]

# ── Load models + data ────────────────────────────────────────
print("Loading models...")
with open("fmow_models/models.pkl", "rb") as f:
    data = pickle.load(f)

svm        = data["svm"]
rf         = data["rf"]
X_test     = data["X_test"]
y_test     = data["y_test"]
X_external = data["X_external"]

# ── Evaluate on test set ──────────────────────────────────────
def evaluate(model, X, y, name):
    y_pred = model.predict(X)
    return {
        "name"      : name,
        "y_pred"    : y_pred,
        "accuracy"  : accuracy_score(y, y_pred),
        "precision" : precision_score(y, y_pred, zero_division=0),
        "recall"    : recall_score(y, y_pred,    zero_division=0),
        "f1"        : f1_score(y, y_pred,        zero_division=0),
        "cm"        : confusion_matrix(y, y_pred),
    }

svm_results = evaluate(svm, X_test, y_test, "SVM")
rf_results  = evaluate(rf,  X_test, y_test, "Random Forest")

for r in [svm_results, rf_results]:
    print(f"\n── {r['name']} — Test Set ──────────────────")
    print(f"  Accuracy  : {r['accuracy']:.4f}")
    print(f"  Precision : {r['precision']:.4f}")
    print(f"  Recall    : {r['recall']:.4f}")
    print(f"  F1        : {r['f1']:.4f}")
    print(f"\n{classification_report(y_test, r['y_pred'], target_names=LABELS)}")

# ── Figure 1: Metrics bar chart ───────────────────────────────
metrics      = ["Accuracy", "Precision", "Recall", "F1"]
svm_scores   = [svm_results[m.lower()] for m in metrics]
rf_scores    = [rf_results[m.lower()]  for m in metrics]

x     = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, svm_scores, width, label="SVM",
               color="steelblue", alpha=0.85)
bars2 = ax.bar(x + width/2, rf_scores,  width, label="Random Forest",
               color="darkorange", alpha=0.85)

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.set_title("Test Set Performance — SVM vs Random Forest")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "test_metrics.png", dpi=150)
plt.show()

# ── Figure 2: Confusion matrices ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, result in zip(axes, [svm_results, rf_results]):
    cm   = result["cm"]
    # Normalize for percentages
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    sns.heatmap(cm_n, annot=False, fmt=".1f", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, ax=ax,
                linewidths=0.5, linecolor="gray", cbar=True)

    # Annotate with count + percentage
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.5,
                    f"{cm[i,j]}\n({cm_n[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=12,
                    color="white" if cm_n[i,j] > 50 else "black",
                    fontweight="bold")

    ax.set_title(f"{result['name']} — Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150)
plt.show()

# ── Figure 3: Per-class performance ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, result in zip(axes, [svm_results, rf_results]):
    cm = result["cm"]
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    bars = ax.bar(LABELS, per_class_acc,
                  color=["steelblue", "forestgreen"], alpha=0.85, width=0.4)
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(f"{result['name']} — Per-Class Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "per_class_accuracy.png", dpi=150)
plt.show()

# ── External image prediction ─────────────────────────────────
print("\n── External Image Prediction ───────────────────")
svm_pred     = svm.predict(X_external)[0]
svm_proba    = svm.predict_proba(X_external)[0]
rf_pred      = rf.predict(X_external)[0]
rf_proba     = rf.predict_proba(X_external)[0]

svm_label    = LABELS[svm_pred]
rf_label     = LABELS[rf_pred]

print(f"  SVM          → {svm_label} (Urban: {svm_proba[0]:.3f} | Natural: {svm_proba[1]:.3f})")
print(f"  Random Forest→ {rf_label} (Urban: {rf_proba[0]:.3f} | Natural: {rf_proba[1]:.3f})")

# Plot external prediction
fig, ax = plt.subplots(figsize=(8, 4))
x     = np.arange(2)
width = 0.3
ax.bar(x - width/2, svm_proba,  width, label="SVM",           color="steelblue",  alpha=0.85)
ax.bar(x + width/2, rf_proba,   width, label="Random Forest", color="darkorange", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.set_ylabel("Probability")
ax.set_ylim(0, 1.15)
ax.set_title("External Image — Prediction Probabilities")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

for i, (sp, rp) in enumerate(zip(svm_proba, rf_proba)):
    ax.text(i - width/2, sp + 0.02, f"{sp:.3f}", ha="center", fontsize=10)
    ax.text(i + width/2, rp + 0.02, f"{rp:.3f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "external_prediction.png", dpi=150)
plt.show()

# ── Summary ───────────────────────────────────────────────────
print(f"""
✅ Part 6 Complete!

Test Set Results:
                  SVM          Random Forest
  Accuracy  :  {svm_results['accuracy']:.4f}         {rf_results['accuracy']:.4f}
  Precision :  {svm_results['precision']:.4f}         {rf_results['precision']:.4f}
  Recall    :  {svm_results['recall']:.4f}         {rf_results['recall']:.4f}
  F1        :  {svm_results['f1']:.4f}         {rf_results['f1']:.4f}

External Image:
  SVM          → {svm_label} (confidence: {max(svm_proba):.3f})
  Random Forest→ {rf_label} (confidence: {max(rf_proba):.3f})

Saved to fmow_evaluation/
  ├── test_metrics.png
  ├── confusion_matrices.png
  ├── per_class_accuracy.png
  └── external_prediction.png
""")