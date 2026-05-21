
# Part 7 — Analysis & Discussion

## 1. Overall Model Performance

| Metric    | SVM (CV) | SVM (Test) | RF (CV)  | RF (Test) |
|-----------|----------|------------|----------|-----------|
| Accuracy  | 0.7670   | 0.7250     | 0.7161   | 0.6821    |
| Precision | 0.7724   | 0.7405     | 0.7035   | 0.6977    |
| Recall    | 0.7571   | 0.6929     | 0.7482   | 0.6429    |
| F1        | 0.7646   | 0.7159     | 0.7249   | 0.6691    |

Both models show a consistent drop from CV to test set (~4% for SVM,
~4-5% for RF), which is expected and indicates no severe overfitting —
just a realistic generalization gap.

---

## 2. Most Discriminative Feature Combinations

The feature pipeline used:
  - Intensity Histogram (64 features) — captures tonal distribution
  - LBP Texture (64 features)         — captures micro-texture patterns
  - Sobel + Canny Edges (18 features) — captures structural complexity
  - HOG Descriptors (324 features)    — captures shape and gradient layout

**LBP + HOG is the most discriminative combination** for this task:
- Urban scenes have high HOG responses due to strong orthogonal edges
  from buildings, roads, and grid structures.
- Natural scenes have high LBP entropy due to irregular organic textures
  like tree canopies, water ripples, and crop patterns.
- Intensity histograms alone are weak — both classes can share similar
  brightness ranges depending on lighting conditions.
- Canny edge ratio is a strong single feature: urban images consistently
  have a higher edge pixel density than natural ones.

---

## 3. SVM Performance Trends

- SVM (RBF kernel, C=10) outperformed Random Forest on every metric.
- CV accuracy of 76.7% dropped to 72.5% on the test set — a 4.2% gap.
- Precision (0.74) > Recall (0.69): SVM is more conservative —
  it prefers not to label something Urban unless confident.
- SVM confidence on the external Urban image was very high (0.971),
  suggesting the RBF kernel found a strong decision boundary for
  structured urban features in PCA space.
- The RBF kernel is well suited here because PCA-reduced features
  are not linearly separable — the kernel projects them into a space
  where the margin is cleaner.

---

## 4. Random Forest Performance Trends

- RF accuracy of 71.6% (CV) dropped to 68.2% (test) — a 4.9% gap.
- RF recall (0.64) << SVM recall (0.69): RF misses more Natural images,
  likely because individual trees overfit to specific texture patterns
  in training crops that don't generalize well.
- RF confidence on external image was lower (0.655 Urban vs 0.345 Natural)
  compared to SVM (0.971), showing RF is less decisive on unseen data.
- With 200 trees and no max depth, RF likely memorized some training
  samples — deeper regularization (max_depth=10-15) may help.

---

## 5. Misclassifications & Limitations

**Why misclassifications happen:**
- Urban categories like "interchange" and "road_bridge" photographed
  in rural settings can look Natural due to surrounding greenery.
- Natural categories like "golf_course" and "surface_mine" have highly
  structured, geometric patterns that resemble Urban scenes to HOG.
- Handcrafted features capture local statistics but miss global context
  — a forest patch in a city looks identical to a rural forest patch.
- All images are 128×128 grayscale — colour information (green for
  vegetation, grey for concrete) is lost, reducing discriminability.
- fMoW images are multi-temporal — the same location photographed under
  different seasons/lighting produces inconsistent feature vectors.

**Class-level observations:**
- Both models perform symmetrically on Urban vs Natural (~1-2% difference),
  meaning neither class is systematically harder — the confusion is spread.
- 140 test samples per class is relatively small for reliable evaluation;
  results may shift with more test data.

---

## 6. Recommendations for Future Hybrid Models

1. **Add colour features** — RGB histograms per channel or HSV histograms
   would immediately improve separation (green vs grey tones).

2. **CNN feature extraction** — Replace handcrafted features with a
   pretrained ResNet/EfficientNet backbone (frozen weights) and use the
   penultimate layer embeddings as features for SVM/RF. This hybrid
   approach typically pushes accuracy above 90%.

3. **Deeper RF regularization** — Set max_depth=10-15 and
   min_samples_leaf=5 to reduce overfitting without losing recall.

4. **SVM hyperparameter tuning** — GridSearchCV over C=[1,10,100] and
   gamma=[scale, auto, 0.01] may squeeze another 2-3% accuracy.

5. **Data augmentation** — Horizontal/vertical flips and random crops
   during training would double effective dataset size cheaply.

6. **SIFT/ORB bag-of-words** — Encoding local keypoint descriptors into
   a visual vocabulary (Fisher vectors or VLAD) would add powerful
   mid-level spatial features not captured by global HOG.

7. **Larger dataset** — 500 images per class is modest. Scaling to
   2000+ per class would likely close the CV-to-test generalization gap.
