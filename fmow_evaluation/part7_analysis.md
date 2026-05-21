
# Satellite Image Classification — Results Report
**Generated:** 2026-05-21 11:15:08

---

## 1. Overall Model Performance

### 5-Fold Cross Validation
| Metric    | SVM                                                        | Random Forest                                             |
|-----------|------------------------------------------------------------|-----------------------------------------------------------|
| Accuracy  | 0.7348 ± 0.0264   | 0.5687 ± 0.0189  |
| Precision | 0.7388 ± 0.0342 | 0.5703 ± 0.0201|
| Recall    | 0.7286 ± 0.0223       | 0.5625 ± 0.0204     |
| F1        | 0.7334 ± 0.0243               | 0.5660 ± 0.0152             |

### Test Set
| Metric    | SVM       | Random Forest |
|-----------|-----------|---------------|
| Accuracy  | 0.7036  | 0.5821        |
| Precision | 0.8065  | 0.6186        |
| Recall    | 0.5357  | 0.4286        |
| F1        | 0.6438  | 0.5063        |

---

## 2. External Image Prediction
| Model         | Prediction       | Urban Prob | Natural Prob |
|---------------|------------------|------------|--------------|
| SVM           | Urban            | 0.968      | 0.032         |
| Random Forest | Urban            | 0.595      | 0.405         |

---

## 3. Feature Pipeline Summary
| Step                | Detail                              | Features |
|---------------------|-------------------------------------|----------|
| Intensity Histogram | 64-bin normalized                   | 64       |
| LBP Texture         | Uniform LBP, P=8, R=1               | 64       |
| Sobel + Canny       | Sobel histogram + edge ratio + mag  | 18       |
| HOG Descriptors     | 9 orientations, 16×16 cells, 2×2    | 324      |
| Raw Total           |                                     | 470      |
| After Var.Threshold | threshold=0.01                      | 8191      |
| After PCA (95% var) | n_components                        | 1010      |

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
