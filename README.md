# Satellite Image Classification (Urban vs. Natural) Using Classical Vision + ML
Build a classical machine-learning system that classifies satellite images into Urban or Natural categories using handcrafted image features. The system must incorporate pixellevel, texture, and gradient-based features and evaluate performance on a real external image.



---
 
## Task Checklist
 
### Part 1 — Dataset Preparation
- [x] Download fMoW dataset from AWS S3
- [x] Select 700 images for **Urban** class
- [x] Select 700 images for **Natural** class
- [x] Create 80:20 train-test split
- [x] Keep one external real-world image for final evaluation
---
 
### Part 2 — Preprocessing
- [ ] Resize each image to 128×128
- [ ] Convert to grayscale
- [ ] Apply Histogram Equalization / CLAHE
- [ ] Apply Gaussian or Median filtering
---
 
### Part 3 — Feature Engineering
- [ ] Pixel intensity histogram (low-level)
- [ ] LBP texture descriptor (low-level)
- [ ] Edge detection using Sobel / Canny (low-level)
- [ ] Histogram of Oriented Gradients — HOG (mid-level)
- [ ] SIFT / ORB descriptors (mid-level, optional)
- [ ] Store extracted features in CSV / NumPy / Pickle format
---
 
### Part 4 — Feature Selection
- [ ] Apply PCA for dimensionality reduction, OR
- [ ] Apply Variance Thresholding to drop low-information features
---
 
### Part 5 — Model Training
- [ ] Train SVM with 5-fold cross-validation
- [ ] Train Random Forest with 5-fold cross-validation
---
 
### Part 6 — Model Evaluation
- [ ] Compute Accuracy, Precision, Recall, F1-score
- [ ] Generate Confusion Matrix
- [ ] Run both models on external real-world image
- [ ] Visualize results — bar charts + confusion matrix heatmap
---
 
### Part 7 — Analysis & Discussion
- [ ] Identify most discriminative feature combinations
- [ ] Discuss SVM performance trends
- [ ] Analyse misclassifications and limitations
- [ ] Recommendations for future hybrid models (handcrafted + CNNs)
---
 
## Dataset Structure
 
```
fmow_data/
├── urban/               (500 images)
├── natural/             (500 images)
├── external/            (1 image — held out for final evaluation)
└── splits/
    ├── train/
    │   ├── urban/       (400 images)
    │   └── natural/     (400 images)
    └── test/
        ├── urban/       (100 images)
        └── natural/     (100 images)
```
 
## Urban Categories (from fMoW)
`multi-unit_residential`, `single-unit_residential`, `office_building`, `shopping_mall`, `parking_lot_or_garage`, `interchange`, `road_bridge`, `ground_transportation_station`
 
## Natural Categories (from fMoW)
`crop_field`, `lake_or_pond`, `park`, `aquaculture`, `golf_course`, `wind_farm`, `surface_mine`, `forest`
 
---