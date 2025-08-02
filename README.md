# MLOps Major Assignment – Linear Regression Pipeline

## Description

This project implements a complete MLOps pipeline using **Scikit-learn's LinearRegression** model trained on the **California Housing dataset**. The project demonstrates:

- Model training using `train.py`
- Manual quantization of model parameters using `quantize.py` (uint8)
- Inference using the dequantized model in `predict.py`
- Unit testing with `pytest`
- Dockerization using `Dockerfile`
- Full CI/CD automation using GitHub Actions

All development is done in a **single `main` branch**, with proper file structuring and GitHub CLI-based submission.

---

## Directory Structure

mlops-major-assignment/
├── src/
│ ├── train.py
│ ├── quantize.py
│ └── predict.py
├── tests/
│ └── test_train.py
├── artifacts/
├── requirements.txt
├── Dockerfile
├── .github/workflows/ci.yml
└── README.md


---

## 🔧 CI/CD Workflow Overview

| Job                    | Description                                      |
|------------------------|--------------------------------------------------|
| `test-suite`           | Runs unit tests using `pytest`                  |
| `train-and-quantize`   | Trains the model and applies manual quantization |
| `build-and-test-container` | Builds Docker image and verifies predictions |

---

## Mandatory Comparison Table

| Metric                | Original Model     | Quantized Model     |
|-----------------------|--------------------|----------------------|
| R² Score              | 0.6053             | -0.1542              |
| File Size (joblib)    | 0.40 KB            | 0.41 KB              |

> Note: Manual quantization introduces accuracy loss but helps demonstrate trade-offs in model compression.

---

## Run Locally

# Train the model
python src/train.py

# Quantize and compare
python src/quantize.py

# Run prediction
python src/predict.py

---

## Run using Docker
docker build -t mlops-predict .
docker run --rm mlops-predict