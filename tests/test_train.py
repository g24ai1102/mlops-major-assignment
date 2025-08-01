# tests/test_train.py

import pytest
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

@pytest.fixture
def data():
    X, y = fetch_california_housing(return_X_y=True)
    return X, y

def test_model_type():
    model = joblib.load("artifacts/model.joblib")
    assert isinstance(model, LinearRegression)

def test_model_trained():
    model = joblib.load("artifacts/model.joblib")
    assert hasattr(model, "coef_"), "Model does not have coefficients. Seems untrained."

def test_r2_score_above_threshold(data):
    X, y = data
    model = joblib.load("artifacts/model.joblib")
    y_pred = model.predict(X)
    score = r2_score(y, y_pred)
    assert score > 0.5, f"R2 score too low: {score}"