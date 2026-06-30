import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROBUSTNESS_GATE_PATH = (
    PROJECT_ROOT / "steps" / "finalCheck" / "robustness_gate.py"
)

sys.path.append(str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "robustness_gate",
    ROBUSTNESS_GATE_PATH,
)
robustness_gate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(robustness_gate_module)


class DummyModel:
    def __init__(self, predictions):
        self.predictions = predictions
        self.call_count = 0

    def predict(self, X):
        prediction = self.predictions[self.call_count]
        self.call_count += 1
        return prediction


def create_test_data():
    X_test = pd.DataFrame(
        {
            "duration": [12, 24, 36, 48],
            "credit_amount": [1000, 2000, 3000, 4000],
            "monthly_payment": [83.33, 83.33, 83.33, 83.33],
        }
    )

    y_test = pd.Series([1, 1, 0, 0])

    return X_test, y_test


def run_robustness_gate(model, X_test, y_test, test_metrics):
    return robustness_gate_module.robustness_gate.entrypoint(
        model,
        X_test,
        y_test,
        test_metrics,
    )


def patch_test_data_generators(monkeypatch):
    def fake_generate_perturbation(X_test):
        transformed = X_test.copy()
        transformed["test_type"] = "perturbation"
        return transformed

    def fake_generate_inflation_test_data(X_test):
        transformed = X_test.copy()
        transformed["test_type"] = "inflation"
        return transformed

    monkeypatch.setattr(
        robustness_gate_module,
        "generate_perturbation",
        fake_generate_perturbation,
    )

    monkeypatch.setattr(
        robustness_gate_module,
        "generate_inflation_test_data",
        fake_generate_inflation_test_data,
    )


def test_robustness_gate_returns_report_and_boolean(monkeypatch):
    X_test, y_test = create_test_data()
    test_metrics = {"f1": 1.0}

    patch_test_data_generators(monkeypatch)

    model = DummyModel(
        predictions=[
            np.array([1, 1, 0, 0]),
            np.array([1, 1, 0, 0]),
        ]
    )

    robustness_report, gate_passed = run_robustness_gate(
        model,
        X_test,
        y_test,
        test_metrics,
    )

    assert isinstance(robustness_report, dict)
    assert isinstance(gate_passed, bool)


def test_robustness_gate_report_has_expected_structure(monkeypatch):
    X_test, y_test = create_test_data()
    test_metrics = {"f1": 1.0}

    patch_test_data_generators(monkeypatch)

    model = DummyModel(
        predictions=[
            np.array([1, 1, 0, 0]),
            np.array([1, 1, 0, 0]),
        ]
    )

    robustness_report, _ = run_robustness_gate(
        model,
        X_test,
        y_test,
        test_metrics,
    )

    assert robustness_report["gate_name"] == "Robustness Quality Gate"
    assert "gate_passed" in robustness_report
    assert "metrics" in robustness_report


def test_robustness_gate_passes_when_mean_score_is_above_threshold(monkeypatch):
    X_test, y_test = create_test_data()
    test_metrics = {"f1": 1.0}

    patch_test_data_generators(monkeypatch)

    model = DummyModel(
        predictions=[
            np.array([1, 1, 0, 0]),
            np.array([1, 1, 0, 0]),
        ]
    )

    robustness_report, gate_passed = run_robustness_gate(
        model,
        X_test,
        y_test,
        test_metrics,
    )

    assert gate_passed is True
    assert robustness_report["gate_passed"] is True
    assert (
        robustness_report["metrics"]["mean_robustness_score"]["passed"]
        is True
    )


def test_robustness_gate_fails_when_mean_score_is_below_threshold(monkeypatch):
    X_test, y_test = create_test_data()
    test_metrics = {"f1": 1.0}

    patch_test_data_generators(monkeypatch)

    model = DummyModel(
        predictions=[
            np.array([0, 0, 0, 0]),
            np.array([0, 0, 0, 0]),
        ]
    )

    robustness_report, gate_passed = run_robustness_gate(
        model,
        X_test,
        y_test,
        test_metrics,
    )

    assert gate_passed is False
    assert robustness_report["gate_passed"] is False
    assert (
        robustness_report["metrics"]["mean_robustness_score"]["passed"]
        is False
    )


def test_robustness_gate_calculates_f1_scores_correctly(monkeypatch):
    X_test, y_test = create_test_data()
    test_metrics = {"f1": 1.0}

    patch_test_data_generators(monkeypatch)

    model = DummyModel(
        predictions=[
            np.array([1, 1, 0, 0]),
            np.array([1, 0, 0, 0]),
        ]
    )

    robustness_report, _ = run_robustness_gate(
        model,
        X_test,
        y_test,
        test_metrics,
    )

    assert robustness_report["metrics"]["f1_clean"] == pytest.approx(1.0)
    assert robustness_report["metrics"]["f1_perturbation"] == pytest.approx(1.0)
    assert robustness_report["metrics"]["f1_inflation"] == pytest.approx(
        2 / 3
    )


def test_robustness_gate_calculates_robustness_scores_correctly(monkeypatch):
    X_test, y_test = create_test_data()
    test_metrics = {"f1": 1.0}

    patch_test_data_generators(monkeypatch)

    model = DummyModel(
        predictions=[
            np.array([1, 1, 0, 0]),
            np.array([1, 0, 0, 0]),
        ]
    )

    robustness_report, _ = run_robustness_gate(
        model,
        X_test,
        y_test,
        test_metrics,
    )

    assert robustness_report["metrics"]["robustness_perturbation"] == pytest.approx(
        1.0
    )
    assert robustness_report["metrics"]["robustness_inflation"] == pytest.approx(
        2 / 3
    )
    assert robustness_report["metrics"]["mean_robustness_score"]["value"] == pytest.approx(
        (1.0 + (2 / 3)) / 2
    )


def test_robustness_gate_uses_threshold_095(monkeypatch):
    X_test, y_test = create_test_data()
    test_metrics = {"f1": 1.0}

    patch_test_data_generators(monkeypatch)

    model = DummyModel(
        predictions=[
            np.array([1, 1, 0, 0]),
            np.array([1, 1, 0, 0]),
        ]
    )

    robustness_report, _ = run_robustness_gate(
        model,
        X_test,
        y_test,
        test_metrics,
    )

    mean_score = robustness_report["metrics"]["mean_robustness_score"]

    assert mean_score["threshold"] == 0.95
    assert mean_score["comparison"] == ">="


def test_robustness_gate_calls_model_predict_twice(monkeypatch):
    X_test, y_test = create_test_data()
    test_metrics = {"f1": 1.0}

    patch_test_data_generators(monkeypatch)

    model = DummyModel(
        predictions=[
            np.array([1, 1, 0, 0]),
            np.array([1, 1, 0, 0]),
        ]
    )

    _ = run_robustness_gate(
        model,
        X_test,
        y_test,
        test_metrics,
    )

    assert model.call_count == 2


def test_robustness_gate_does_not_modify_original_x_test(monkeypatch):
    X_test, y_test = create_test_data()
    original_X_test = X_test.copy(deep=True)
    test_metrics = {"f1": 1.0}

    patch_test_data_generators(monkeypatch)

    model = DummyModel(
        predictions=[
            np.array([1, 1, 0, 0]),
            np.array([1, 1, 0, 0]),
        ]
    )

    _ = run_robustness_gate(
        model,
        X_test,
        y_test,
        test_metrics,
    )

    pd.testing.assert_frame_equal(X_test, original_X_test)


def test_robustness_gate_contains_all_expected_metrics(monkeypatch):
    X_test, y_test = create_test_data()
    test_metrics = {"f1": 1.0}

    patch_test_data_generators(monkeypatch)

    model = DummyModel(
        predictions=[
            np.array([1, 1, 0, 0]),
            np.array([1, 1, 0, 0]),
        ]
    )

    robustness_report, _ = run_robustness_gate(
        model,
        X_test,
        y_test,
        test_metrics,
    )

    expected_metrics = {
        "f1_clean",
        "f1_perturbation",
        "f1_inflation",
        "robustness_perturbation",
        "robustness_inflation",
        "mean_robustness_score",
    }

    assert set(robustness_report["metrics"].keys()) == expected_metrics