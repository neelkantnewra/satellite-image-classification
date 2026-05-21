
# Satellite Image Classification — Results Report
**Generated:** 2026-05-21 11:34:50

---

## 1. Overall Model Performance

### 5-Fold Cross Validation
| Metric    | SVM                                                        | Random Forest                                             |
|-----------|------------------------------------------------------------|-----------------------------------------------------------|
| Accuracy  | 0.7598 ± 0.0268   | 0.7161 ± 0.0249  |
| Precision | 0.7626 ± 0.0173 | 0.7035 ± 0.0261|
| Recall    | 0.7536 ± 0.0488       | 0.7482 ± 0.0273     |
| F1        | 0.7576 ± 0.0325               | 0.7249 ± 0.0233             |

### Test Set
| Metric    | SVM       | Random Forest |
|-----------|-----------|---------------|
| Accuracy  | 0.7429  | 0.6821        |
| Precision | 0.7742  | 0.6977        |
| Recall    | 0.6857  | 0.6429        |
| F1        | 0.7273  | 0.6691        |

---

## 2. External Image Prediction
| Model         | Prediction       | Urban Prob | Natural Prob |
|---------------|------------------|------------|--------------|
| SVM           | Urban            | 0.966      | 0.034         |
| Random Forest | Urban            | 0.655      | 0.345         |

---

## 3. Feature Pipeline Summary
| Step                | Detail                              | Features |
|---------------------|-------------------------------------|----------|
| Intensity Histogram | 64-bin normalized                   | 64       |
| LBP Texture         | Uniform LBP, P=8, R=1               | 64       |
| Sobel + Canny       | Sobel histogram + edge ratio + mag  | 18       |
| HOG Descriptors     | 9 orientations, 16×16 cells, 2×2    | 324      |
| Raw Total           |                                     | 470      |
| After Var.Threshold | threshold=0.01                      | 1855      |
| After PCA (95% var) | n_components                        | 356      |

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
