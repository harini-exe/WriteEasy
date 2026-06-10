# WriteEasy — IMU-Based Handwriting Recognition System for Dyslexia Support

## Overview
WriteEasy is a sensor-integrated assistive writing system designed 
to help children with dyslexia improve handwriting through 
real-time feedback. The system combines a custom-built smart pen, 
a wearable glove, and a machine learning pipeline to capture, 
analyze, and classify handwriting motion in real time.

Built as part of Kurukshetra 2026 inter-college competition — 
**Secured 3rd place**

---

## System Architecture

### Hardware
- **Smart Pen:** Custom SLA 3D-printed pen housing MPU6050 IMU 
  + FSR pressure sensor with spring-based force transmission
- **Wearable Glove:** ESP32 microcontroller + flex sensors on 
  thumb and index finger for grip posture monitoring
- **Custom PCB:** Fabricated to replace breadboard prototype — 
  compact, ruggedized for real movement
- **Data Transmission:** 100 Hz sampling via LittleFS local 
  buffer → asynchronous HTTPS POST to Flask backend

### ML Pipeline
- **Algorithm:** DTW-KNN (Dynamic Time Warping + 
  K-Nearest Neighbours, K=5)
- **Why DTW:** Handwriting sequences vary in speed and length — 
  DTW handles variable-length time series better than 
  Euclidean distance, aligning sequences elastically
- **Features:** 6-axis IMU data — ax, ay, az (accelerometer), 
  gx, gy, gz (gyroscope)
- **Sensor Fusion:** Madgwick filter for drift-free 
  orientation estimation

### Preprocessing Pipeline
1. Pressure threshold filtering (FSR > 0.1V — pen on paper only)
2. Motion energy filtering (removes stationary/idle samples)
3. Resample sequences to fixed length (120 samples via interpolation)
4. Z-score normalization per feature axis
5. Exponential smoothing for noise reduction (α=0.7)
6. DTW comparison with Sakoe-Chiba band constraint (window=20)

### Backend API
- Flask REST API receiving real-time JSON sensor data
- Predicts letter + returns confidence score
- Deployed locally, communicates with web frontend

---

## Results

**Overall Accuracy: 97%** on 180 test samples (A, C, X classes)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| A     | 1.00      | 0.90   | 0.95     | 10      |
| C     | 0.94      | 1.00   | 0.97     | 16      |
| X     | 1.00      | 1.00   | 1.00     | 10      |
| **Weighted Avg** | **0.97** | **0.97** | **0.97** | **36** |

**Confusion Matrix:** Only 1 misclassification — 
1 instance of A predicted as C. Classes A/C show minor 
overlap due to similar stroke curvature in DTW feature space. 
Class X achieved perfect classification (zero confusion).

---

## Key Findings
- DTW outperforms Euclidean distance for this task because 
  handwriting speed varies significantly between trials — 
  elastic alignment captures stroke shape regardless of duration
- A/C confusion is expected given their shared curved stroke 
  component — future work: expand training set and add 
  CNN-based feature extraction
- Template-based DTW-KNN requires no GPU training, 
  making it deployable on edge devices with minimal compute

---

## Tech Stack
Python, NumPy, pandas, scikit-learn, Flask, matplotlib,  
ESP32 (Arduino/PlatformIO), MPU6050, HTML5/CSS3/JavaScript

## My Role
Team Lead 
- Directed model selection and evaluation strategy
- Evaluated DTW-KNN, LSTM, Decision Tree approaches
  using confusion matrices and accuracy metrics  
- Selected DTW-KNN based on performance analysis 
  of time-series classification results
- Supervised software implementation and system integration
- Led end-to-end pipeline from data acquisition to 
  real-time API deployment
