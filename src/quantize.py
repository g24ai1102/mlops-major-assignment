# src/quantize.py

import joblib
import numpy as np
import os
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

# Load model
model = joblib.load("artifacts/model.joblib")
coef = model.coef_
intercept = model.intercept_

# Save original params
os.makedirs("artifacts", exist_ok=True)
joblib.dump({"coef": coef, "intercept": intercept}, "artifacts/unquant_params.joblib")
print("Saved unquantized parameters")

# Quantize only coef to uint8 using min-max scaling
coef_min = coef.min()
coef_max = coef.max()
quantized_coef = np.round((coef - coef_min) / (coef_max - coef_min) * 255).astype(np.uint8)

# Save quantized coef + intercept + scaling info
joblib.dump({
    "coef": quantized_coef,
    "intercept": intercept,  # leave intercept unquantized
    "coef_min": coef_min,
    "coef_max": coef_max
}, "artifacts/quant_params.joblib")
print("Saved quantized parameters")

# Dequantize coef
dequantized_coef = quantized_coef.astype(np.float32) / 255 * (coef_max - coef_min) + coef_min

# Predict using dequantized coef and original intercept
X, y = fetch_california_housing(return_X_y=True)
y_pred = X @ dequantized_coef + intercept
r2 = r2_score(y, y_pred)
print(f"R2 after quantization: {r2:.4f}")