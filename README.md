# WriteEasy — ML Pipeline for IMU-Based Handwriting Recognition

This repository contains my individual software contribution 
to WriteEasy, a team project developed for Kurukshetra 2026.

## Overview
WriteEasy is a sensor-integrated assistive writing system for 
children with dyslexia. This repository contains the ML pipeline 
component — DTW-KNN based handwriting recognition using 
6-axis IMU sensor data captured from a custom smart pen and 
wearable glove built by the hardware team.

**Kurukshetra 2026 inter-college competition — 3rd place**

---

## ML Pipeline

**Algorithm:** DTW-KNN (Dynamic Time Warping + K-Nearest Neighbours, K=5)

**Why DTW:** Handwriting sequences vary in speed and length — 
DTW handles variable-length time series better than Euclidean 
distance, aligning sequences elastically across time.

**Features:** 6-axis IMU data — ax, ay, az (accelerometer), 
gx, gy, gz (gyroscope)

---

## Preprocessing Pipeline

1. Pressure threshold filtering — pen-on-paper samples only
2. Motion energy filtering — removes stationary/idle samples
3. Resample sequences to fixed length (120 samples via interpolation)
4. Z-score normalization per feature axis
5. Exponential smoothing for noise reduction (α=0.7)
6. DTW comparison with Sakoe-Chiba band constraint (window=20)

---

## Results

**Overall Accuracy: 97%** on 36 test samples (179 total training 
samples across A, C, X classes)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| A | 1.00 | 0.90 | 0.95 | 10 |
| C | 0.94 | 1.00 | 0.97 | 16 |
| X | 1.00 | 1.00 | 1.00 | 10 |
| **Weighted Avg** | **0.97** | **0.97** | **0.97** | **36** |

**Confusion Matrix:** Only 1 misclassification — 1 instance of A 
predicted as C. Classes A/C show minor overlap due to similar 
stroke curvature in DTW feature space. Class X achieved perfect 
classification (zero confusion).

---

## Key Findings

- DTW outperforms Euclidean distance for this task because 
  handwriting speed varies significantly between trials — 
  elastic alignment captures stroke shape regardless of duration
- A/C confusion is expected given their shared curved stroke 
  component — future work: expand training set and add 
  CNN-based feature extraction
- Template-based DTW-KNN requires no GPU training, making 
  it deployable on edge devices with minimal compute

---

## Tech Stack
Python, NumPy, pandas, scikit-learn, matplotlib

---

## My Role
**Team Lead (Software Component)**
- Directed model selection and evaluation strategy
- Evaluated DTW-KNN, LSTM, Decision Tree approaches 
  using confusion matrices and accuracy metrics
- Selected DTW-KNN based on performance analysis of 
  time-series classification results
- Supervised software implementation and system integration

---

## Note
Training dataset not included in this repository. 
