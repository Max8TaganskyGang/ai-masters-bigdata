#!/usr/bin/env python3
import sys
import joblib
import numpy as np
import ast

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "2a.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

def predict(features):
    try:
        features_array = np.array(ast.literal_eval(features)).reshape(1, -1)
        prediction = model.predict_proba(features_array)[0][1]
        return float(prediction)
    except Exception as e:
        return float(0)

for line in sys.stdin:
    print(predict(line.strip()))
