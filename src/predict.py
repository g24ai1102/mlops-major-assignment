# src/predict.py

import joblib
from sklearn.datasets import fetch_california_housing

# Load trained model
model = joblib.load("artifacts/model.joblib")

# Load dataset
X, y = fetch_california_housing(return_X_y=True)

# Predict first 5 samples
y_pred = model.predict(X[:5])

# Print predictions with flush=True to force Docker to print output
print("🔮 Sample predictions:", flush=True)
for i, pred in enumerate(y_pred):
    print(f"Prediction {i+1}: {pred:.2f}", flush=True)