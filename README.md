<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=32&duration=3000&pause=1000&color=FFFFFF&background=1a472a&center=true&vCenter=true&width=700&height=100&lines=🛰️+Satellite+Image+Classification;Urban+vs+Natural+%7C+fMoW+Dataset" alt="Typing SVG" />

  <p><strong>Classical Computer Vision + Machine Learning Pipeline</strong><br></p>

  ![Last Commit](https://img.shields.io/github/last-commit/neelkantnewra/satellite-image-classification?color=green&style=flat-square)
  ![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)
  ![Python](https://img.shields.io/badge/python-3.9+-blue?style=flat-square&logo=python)
  ![Dataset](https://img.shields.io/badge/dataset-fMoW-orange?style=flat-square)
  ![SVM Accuracy](https://img.shields.io/badge/SVM%20accuracy-72.5%25-brightgreen?style=flat-square)
  ![RF Accuracy](https://img.shields.io/badge/RF%20accuracy-68.2%25-yellow?style=flat-square)
</div>

---

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
- [x] Resize each image to 128×128
- [x] Convert to grayscale
- [x] Apply Histogram Equalization / CLAHE
- [x] Apply Gaussian or Median filtering
---
 
### Part 3 — Feature Engineering
- [x] Pixel intensity histogram (low-level)
- [x] LBP texture descriptor (low-level)
- [x] Edge detection using Sobel / Canny (low-level)
- [x] Histogram of Oriented Gradients — HOG (mid-level)
- [x] SIFT / ORB descriptors (mid-level, optional)
- [x] Store extracted features in CSV / NumPy / Pickle format
---
 
### Part 4 — Feature Selection
- [x] Apply PCA for dimensionality reduction, OR
- [x] Apply Variance Thresholding to drop low-information features
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

---

## Getting Started
 
### Prerequisites
- Python 3.9+
- AWS CLI (for dataset download)
- ~500 MB free disk space
### Step 1 — Clone the repository
```bash
git clone https://github.com/neelkantnewra/satellite-image-classification.git
cd satellite-image-classification
```
 
### Step 2 — Create virtual environment
```bash
python3 -m venv .venv
```
 
### Step 3 — Activate virtual environment
```bash
# Mac / Linux
source .venv/bin/activate
 
# Windows
.venv\Scripts\activate
```
 
### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```
 
### Step 5 — Run the full pipeline
```bash
make all
```

---

 
## Dataset Structure
 
```
fmow_data/
├── urban/               (700 images)
├── natural/             (700 images)
├── external/            (1 image — held out for final evaluation)
└── splits/
    ├── train/
    │   ├── urban/       (560 images)
    │   └── natural/     (560 images)
    └── test/
        ├── urban/       (140 images)
        └── natural/     (140 images)
```
 
## Urban Categories (from fMoW)
`multi-unit_residential`, `single-unit_residential`, `office_building`, `shopping_mall`, `parking_lot_or_garage`, `interchange`, `road_bridge`, `ground_transportation_station`
 
## Natural Categories (from fMoW)
`crop_field`, `lake_or_pond`, `park`, `aquaculture`, `golf_course`, `wind_farm`, `surface_mine`, `forest`
 
---