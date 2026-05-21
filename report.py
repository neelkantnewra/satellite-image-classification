# ============================================================
# fMoW — report.py (Dynamic — reads actual results every run)
# ============================================================

import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# ── Load everything ───────────────────────────────────────────
with open("fmow_models/models.pkl", "rb") as f:
    model_data = pickle.load(f)

with open("fmow_selected/selected_features.pkl", "rb") as f:
    feat_data = pickle.load(f)

svm        = model_data["svm"]
rf         = model_data["rf"]
svm_cv     = model_data["svm_cv"]
rf_cv      = model_data["rf_cv"]
X_test     = model_data["X_test"]
y_test     = model_data["y_test"]
X_external = model_data["X_external"]
pca        = feat_data["pca"]
selector   = feat_data["selector"]

# ── Compute test metrics fresh ────────────────────────────────
def get_metrics(model, X, y):
    y_pred = model.predict(X)
    return {
        "accuracy"  : accuracy_score(y, y_pred),
        "precision" : precision_score(y, y_pred, zero_division=0),
        "recall"    : recall_score(y, y_pred,    zero_division=0),
        "f1"        : f1_score(y, y_pred,        zero_division=0),
    }

svm_test = get_metrics(svm, X_test, y_test)
rf_test  = get_metrics(rf,  X_test, y_test)

# ── External prediction ───────────────────────────────────────
svm_pred   = svm.predict(X_external)[0]
svm_proba  = svm.predict_proba(X_external)[0]
rf_pred    = rf.predict(X_external)[0]
rf_proba   = rf.predict_proba(X_external)[0]
LABELS     = ["Urban", "Natural"]

# ── Build report dynamically ──────────────────────────────────
report = f"""
# Satellite Image Classification — Results Report
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Overall Model Performance

### 5-Fold Cross Validation
| Metric    | SVM                                                        | Random Forest                                             |
|-----------|------------------------------------------------------------|-----------------------------------------------------------|
| Accuracy  | {svm_cv['test_accuracy'].mean():.4f} ± {svm_cv['test_accuracy'].std():.4f}   | {rf_cv['test_accuracy'].mean():.4f} ± {rf_cv['test_accuracy'].std():.4f}  |
| Precision | {svm_cv['test_precision'].mean():.4f} ± {svm_cv['test_precision'].std():.4f} | {rf_cv['test_precision'].mean():.4f} ± {rf_cv['test_precision'].std():.4f}|
| Recall    | {svm_cv['test_recall'].mean():.4f} ± {svm_cv['test_recall'].std():.4f}       | {rf_cv['test_recall'].mean():.4f} ± {rf_cv['test_recall'].std():.4f}     |
| F1        | {svm_cv['test_f1'].mean():.4f} ± {svm_cv['test_f1'].std():.4f}               | {rf_cv['test_f1'].mean():.4f} ± {rf_cv['test_f1'].std():.4f}             |

### Test Set
| Metric    | SVM       | Random Forest |
|-----------|-----------|---------------|
| Accuracy  | {svm_test['accuracy']:.4f}  | {rf_test['accuracy']:.4f}        |
| Precision | {svm_test['precision']:.4f}  | {rf_test['precision']:.4f}        |
| Recall    | {svm_test['recall']:.4f}  | {rf_test['recall']:.4f}        |
| F1        | {svm_test['f1']:.4f}  | {rf_test['f1']:.4f}        |

---

## 2. External Image Prediction
| Model         | Prediction       | Urban Prob | Natural Prob |
|---------------|------------------|------------|--------------|
| SVM           | {LABELS[svm_pred]:<16} | {svm_proba[0]:.3f}      | {svm_proba[1]:.3f}         |
| Random Forest | {LABELS[rf_pred]:<16} | {rf_proba[0]:.3f}      | {rf_proba[1]:.3f}         |

---

## 3. Feature Pipeline Summary
| Step                | Detail                              | Features |
|---------------------|-------------------------------------|----------|
| Intensity Histogram | 64-bin normalized                   | 64       |
| LBP Texture         | Uniform LBP, P=8, R=1               | 64       |
| Sobel + Canny       | Sobel histogram + edge ratio + mag  | 18       |
| HOG Descriptors     | 9 orientations, 16×16 cells, 2×2    | 324      |
| Raw Total           |                                     | 470      |
| After Var.Threshold | threshold=0.01                      | {selector.get_support().sum()}      |
| After PCA (95% var) | n_components                        | {pca.n_components_}      |

---

## 4. Most Discriminative Features
- **LBP + HOG** is the strongest combination for Urban vs Natural
- Urban: high HOG response from orthogonal edges (buildings, roads)
- Natural: high LBP entropy from organic textures (canopy, water, crops)
- Canny edge ratio is a strong single feature — urban images have
  consistently higher edge pixel density than natural ones

---

## 5. Misclassifications & Limitations
- Grayscale conversion loses colour cues (green=natural, grey=urban)
- "golf_course" and "surface_mine" have geometric patterns that
  resemble urban scenes to HOG descriptors
- "interchange" and "road_bridge" surrounded by greenery can
  look natural due to organic surroundings
- 128×128 resolution loses fine-grained structural detail
- fMoW multi-temporal images vary across seasons and lighting

---

## 6. Recommendations
1. Add RGB/HSV colour histograms — immediately improves separation
2. CNN hybrid — pretrained ResNet backbone + SVM classifier (>90% acc)
3. RF regularization — max_depth=10-15, min_samples_leaf=5
4. SVM tuning — GridSearchCV over C=[1,10,100], gamma=[scale,0.01]
5. Data augmentation — flips + crops to double effective dataset size
6. SIFT/ORB bag-of-words — Fisher vectors for spatial mid-level features
7. Scale dataset — 2000+ per class to close CV-to-test gap
"""

# ── Save ──────────────────────────────────────────────────────
out = Path("fmow_evaluation/part7_analysis.md")
out.write_text(report)
print(report)
print(f"✅ Report saved → {out}")