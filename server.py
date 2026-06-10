from flask_cors import CORS
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import os
import time
import pipeline

app = Flask(__name__)
CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "TESTINGDATA")

# thresholds
PRESSURE_THRESHOLD = 0.1
MOTION_THRESHOLD = 0.02
MIN_SAMPLES = 30

print("Loading training data...")
training_data = pipeline.load_training_data()
print("Training ready!")


def clear_testing_folder():
    for f in os.listdir(TEST_DIR):
        path = os.path.join(TEST_DIR, f)
        if os.path.isfile(path):
            os.remove(path)


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    if not data:
        return jsonify({"error": "no data received"}), 400

    # 1️⃣ CLEAR OLD TEST FILES
    clear_testing_folder()

    seq = []

    # 2️⃣ FILTER DATA (pressure + motion)
    for r in data:

        pressure = r.get("pressure", 0)

        ax = r.get("ax", 0)
        ay = r.get("ay", 0)
        az = r.get("az", 0)

        gx = r.get("gx", 0)
        gy = r.get("gy", 0)
        gz = r.get("gz", 0)

        timestamp = r.get("timestamp", 0)

        # Ignore if pen not touching paper
        if pressure < PRESSURE_THRESHOLD:
            continue

        # Motion energy
        motion = (ax**2 + ay**2 + az**2) ** 0.5

        # Ignore stationary samples
        if motion < MOTION_THRESHOLD:
            continue

        seq.append([
            timestamp,
            ax,
            ay,
            az,
            gx,
            gy,
            gz
        ])

    # 3️⃣ CHECK MINIMUM SAMPLE LENGTH
    if len(seq) < MIN_SAMPLES:
        print("Too few valid samples:", len(seq))
        return jsonify({
            "letter": "?",
            "confidence": 0.0
        })

    seq = np.array(seq)

    # 4️⃣ SAVE CSV TO TESTINGDATA
    filename = f"test_{int(time.time())}.csv"
    filepath = os.path.join(TEST_DIR, filename)

    df = pd.DataFrame(seq)
    df.to_csv(filepath, index=False, header=False)

    print("Saved new test file:", filename)

    # 5️⃣ PREPARE FEATURES
    seq_features = seq[:, 1:7]

    seq_features = pipeline.resample_sequence(seq_features)
    seq_features = pipeline.normalize(seq_features)
    seq_features = pipeline.smooth(seq_features)

    # 6️⃣ RUN DTW PREDICTION
    label, dist = pipeline.predict_sequence(seq_features, training_data)

    confidence = float(max(0.1, min(1.0, 1 / (1 + dist / 200))))

    print("Prediction:", label)

    return jsonify({
        "letter": label,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)