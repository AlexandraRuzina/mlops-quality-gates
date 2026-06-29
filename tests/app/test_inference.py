import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(APP_DIR))

import inference


@pytest.fixture
def sample_input_data():
    return pd.DataFrame(
        [{
            "duration": 24,
            "credit_amount": 3000,
            "checking_status": "<0",
        }]
    )


class DummyModelWithProba:
    def predict(self, input_data):
        return np.array([1])

    def predict_proba(self, input_data):
        return np.array([[0.25, 0.75]])


class DummyModelWithoutProba:
    def predict(self, input_data):
        return np.array([0])


def test_load_model_sets_tracking_uri_and_loads_model(monkeypatch):
    dummy_model = DummyModelWithProba()
    calls = {}

    def fake_set_tracking_uri(uri):
        calls["tracking_uri"] = uri

    def fake_load_model(model_uri):
        calls["model_uri"] = model_uri
        return dummy_model

    monkeypatch.setattr(
        inference.mlflow,
        "set_tracking_uri",
        fake_set_tracking_uri,
    )

    monkeypatch.setattr(
        inference.mlflow.sklearn,
        "load_model",
        fake_load_model,
    )

    model = inference.load_model()

    assert model is dummy_model
    assert calls["tracking_uri"] == inference.MLFLOW_TRACKING_URI
    assert calls["model_uri"] == inference.MODEL_URI


def test_predict_credit_risk_returns_bad_prediction_with_probabilities(sample_input_data):
    model = DummyModelWithProba()

    result = inference.predict_credit_risk(
        model=model,
        input_data=sample_input_data,
    )

    assert result["prediction"] == 1
    assert result["prediction_label"] == "bad"
    assert result["probability_good"] == 0.25
    assert result["probability_bad"] == 0.75


def test_predict_credit_risk_returns_good_prediction_without_probabilities(sample_input_data):
    model = DummyModelWithoutProba()

    result = inference.predict_credit_risk(
        model=model,
        input_data=sample_input_data,
    )

    assert result["prediction"] == 0
    assert result["prediction_label"] == "good"
    assert "probability_good" not in result
    assert "probability_bad" not in result


def test_predict_credit_risk_converts_numpy_types_to_python_types(sample_input_data):
    model = DummyModelWithProba()

    result = inference.predict_credit_risk(
        model=model,
        input_data=sample_input_data,
    )

    assert isinstance(result["prediction"], int)
    assert isinstance(result["probability_good"], float)
    assert isinstance(result["probability_bad"], float)


def test_predict_credit_risk_probabilities_sum_to_one(sample_input_data):
    model = DummyModelWithProba()

    result = inference.predict_credit_risk(
        model=model,
        input_data=sample_input_data,
    )

    total_probability = (
        result["probability_good"]
        + result["probability_bad"]
    )

    assert total_probability == pytest.approx(1.0)