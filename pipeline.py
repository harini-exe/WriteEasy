"""
WriteEasy — IMU Handwriting Recognition Pipeline
=================================================
Algorithm: Dynamic Time Warping (DTW) + K-Nearest Neighbours (KNN)

Pipeline:
1. Load IMU CSV (ax, ay, az, gx, gy, gz)
2. Resample sequence to fixed length (120 samples)
3. Z-score normalize per feature axis
4. Exponential smoothing for noise reduction
5. DTW distance computation with Sakoe-Chiba band constraint
6. KNN voting (K=5) for final character prediction

Note: Training dataset not included in repository.
"""

import os
import glob
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "TRAINING DATASETS")
TEST_DIR = os.path.join(BASE_DIR, "TESTINGDATA")

RESAMPLE_LEN = 120
K_NEIGHBORS = 5

# ax ay az gx gy gz
FEATURE_COLS = [1,2,3,4,5,6]

# DTW window constraint (speed improvement)
DTW_WINDOW = 20


# ─────────────────────────────────────────────
# CSV LOADER
# ─────────────────────────────────────────────

def load_csv(filepath):

    try:
        df = pd.read_csv(filepath, header=None)

        seq = df.iloc[:,FEATURE_COLS].astype(float).values

        if len(seq) < 5:
            print("⚠ Very short sequence:", os.path.basename(filepath))
            return None

        return seq

    except:
        print("⚠ Skipping bad CSV:", os.path.basename(filepath))
        return None


# ─────────────────────────────────────────────
# RESAMPLING
# ─────────────────────────────────────────────

def resample_sequence(seq, target_len=RESAMPLE_LEN):

    T, F = seq.shape

    old_idx = np.linspace(0, T-1, T)
    new_idx = np.linspace(0, T-1, target_len)

    resampled = np.zeros((target_len, F))

    for f in range(F):

        resampled[:,f] = np.interp(
            new_idx,
            old_idx,
            seq[:,f]
        )

    return resampled


# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────

def normalize(seq):

    mean = np.mean(seq, axis=0)
    std = np.std(seq, axis=0) + 1e-6

    return (seq - mean) / std


# ─────────────────────────────────────────────
# SMOOTHING (NOISE REDUCTION)
# ─────────────────────────────────────────────

def smooth(seq):

    for i in range(1, len(seq)):
        seq[i] = 0.7 * seq[i] + 0.3 * seq[i-1]

    return seq


# ─────────────────────────────────────────────
# DTW DISTANCE (FAST VERSION)
# ─────────────────────────────────────────────

def dtw_distance(s1, s2):

    n = len(s1)
    m = len(s2)

    window = max(DTW_WINDOW, abs(n-m))

    dtw = np.full((n+1, m+1), np.inf)
    dtw[0,0] = 0

    for i in range(1, n+1):

        start = max(1, i-window)
        end = min(m+1, i+window)

        for j in range(start, end):

            cost = np.linalg.norm(s1[i-1] - s2[j-1])

            dtw[i,j] = cost + min(
                dtw[i-1,j],
                dtw[i,j-1],
                dtw[i-1,j-1]
            )

    return dtw[n,m]


# ─────────────────────────────────────────────
# LOAD TRAINING DATA
# ─────────────────────────────────────────────

def load_training_data():

    data = []

    labels = sorted([
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR,d))
    ])

    print("Labels detected:", labels)

    for label in labels:

        folder = os.path.join(RAW_DIR,label)

        files = glob.glob(
            os.path.join(folder,"*.csv")
        )

        print(label, "files:", len(files))

        for f in files:

            seq = load_csv(f)

            if seq is None:
                continue

            seq = resample_sequence(seq)
            seq = normalize(seq)
            seq = smooth(seq)

            data.append((seq,label))

    return data


# ─────────────────────────────────────────────
# DTW KNN PREDICTION
# ─────────────────────────────────────────────

def predict_sequence(seq, training_data):

    distances = []

    for train_seq, label in training_data:

        dist = dtw_distance(seq, train_seq)

        distances.append((dist,label))

    distances.sort(key=lambda x: x[0])

    top_k = distances[:K_NEIGHBORS]

    votes = {}

    for dist,label in top_k:
        votes[label] = votes.get(label,0) + 1

    predicted = max(votes, key=votes.get)

    best_distance = top_k[0][0]

    return predicted, best_distance


# ─────────────────────────────────────────────
# TEST PREDICTIONS
# ─────────────────────────────────────────────

def predict_all(training_data):

    print("\nSTEP 2 — TESTING")

    files = glob.glob(
        os.path.join(TEST_DIR,"*.csv")
    )

    if len(files) == 0:

        print("No test files found in TESTINGDATA")
        return

    for fp in files:

        seq = load_csv(fp)

        if seq is None:
            continue

        seq = resample_sequence(seq)
        seq = normalize(seq)
        seq = smooth(seq)

        pred, dist = predict_sequence(seq, training_data)

        print(
            os.path.basename(fp),
            "→",
            pred,
            "(DTW:", round(dist,2),")"
        )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    print("\nIMU HANDWRITING RECOGNITION (DTW)")

    print("\nSTEP 1 — LOADING TRAINING DATA")

    training_data = load_training_data()

    print("\nTotal training samples:", len(training_data))

    predict_all(training_data)

    print("\n✅ DTW pipeline finished successfully")


if __name__ == "__main__":
    main()
