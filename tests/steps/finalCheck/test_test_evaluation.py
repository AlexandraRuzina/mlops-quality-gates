import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEST_EVALUATION_PATH = (
    PROJECT_ROOT / "steps" / "finalCheck" / "test_evaluation.py"
)

sys.path.append(str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "test_evaluation_module",
    TEST_EVALUATION_PATH,
)
test_evaluation_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_evaluation_module)


class DummyPipelineRun:
    id = "dummy-zenml-run-id"
    name = "dummy-pipeline-run"


class DummyContext:
    pipeline_run = DummyPipelineRun()


class DummyMLflowRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class DummyModelWithProba:
    def predict(self, X):
        return np.array([0, 1, 1, 0])

    def predict_proba(self, X):
        return np.array(
            [
                [0.90, 0.10],
                [0.20, 0.80],
                [0.40, 0.60],
                [0.70, 0.30],
            ]
        )


class DummyModelWithoutProba:
    def predict(self, X):
        return np.array([0, 1, 1, 0])


def create_test_data():
    X_test = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4],
            "feature_2": [10, 20, 30, 40],
        }
    )

    y_test = pd.Series([0, 1, 0, 0])

    return X_test, y_test


def patch_mlflow(monkeypatch):
    calls = {
        "tracking_uri": None,
        "run_id": None,
        "tags": {},
        "metrics": {},
    }

    monkeypatch.setattr(
        test_evaluation_module.mlflow,
        "set_tracking_uri",
        lambda uri: calls.update({"tracking_uri": uri}),
    )

    def fake_start_run(run_id):
        calls["run_id"] = run_id
        return DummyMLflowRun()

    monkeypatch.setattr(
        test_evaluation_module.mlflow,
        "start_run",
        fake_start_run,
    )

    monkeypatch.setattr(
        test_evaluation_module.mlflow,
        "set_tag",
        lambda key, value: calls["tags"].update({key: value}),
    )

    monkeypatch.setattr(
        test_evaluation_module.mlflow,
        "log_metric",
        lambda key, value: calls["metrics"].update({key: value}),
    )

    return calls


def run_test_evaluation(
    model_pipeline,
    X_test,
    y_test,
    model_name="random_forest",
    mlflow_run_id="dummy-mlflow-run-id",
):
    return test_evaluation_module.test_evaluation.entrypoint(
        model_pipeline,
        X_test,
        y_test,
        model_name,
        mlflow_run_id,
    )


def test_test_evaluation_returns_metrics_predictions_and_probabilities(monkeypatch):
    X_test, y_test = create_test_data()
    model = DummyModelWithProba()

    patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        test_evaluation_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    test_metrics, y_pred, y_proba = run_test_evaluation(
        model,
        X_test,
        y_test,
    )

    assert isinstance(test_metrics, dict)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y_proba, np.ndarray)


def test_test_evaluation_calculates_expected_metrics_with_probabilities(monkeypatch):
    X_test, y_test = create_test_data()
    model = DummyModelWithProba()

    patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        test_evaluation_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    test_metrics, _, _ = run_test_evaluation(model, X_test, y_test)

    assert test_metrics["accuracy"] == pytest.approx(0.75)
    assert test_metrics["precision"] == pytest.approx(0.5)
    assert test_metrics["recall"] == pytest.approx(1.0)
    assert test_metrics["f1"] == pytest.approx(2 / 3)
    assert test_metrics["roc_auc"] == pytest.approx(1.0)
    assert test_metrics["mcc"] == pytest.approx(0.5773502692)


def test_test_evaluation_returns_positive_class_probabilities(monkeypatch):
    X_test, y_test = create_test_data()
    model = DummyModelWithProba()

    patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        test_evaluation_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    _, _, y_proba = run_test_evaluation(model, X_test, y_test)

    expected_y_proba = np.array([0.10, 0.80, 0.60, 0.30])

    np.testing.assert_array_equal(y_proba, expected_y_proba)


def test_test_evaluation_handles_model_without_predict_proba(monkeypatch):
    X_test, y_test = create_test_data()
    model = DummyModelWithoutProba()

    patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        test_evaluation_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    test_metrics, y_pred, y_proba = run_test_evaluation(model, X_test, y_test)

    assert isinstance(test_metrics["roc_auc"], float)
    assert np.isnan(test_metrics["roc_auc"])
    assert y_proba is None
    np.testing.assert_array_equal(y_pred, np.array([0, 1, 1, 0]))


def test_test_evaluation_logs_to_existing_mlflow_run(monkeypatch):
    X_test, y_test = create_test_data()
    model = DummyModelWithProba()

    calls = patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        test_evaluation_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    _ = run_test_evaluation(
        model,
        X_test,
        y_test,
        model_name="random_forest",
        mlflow_run_id="existing-run-id",
    )

    assert calls["tracking_uri"] == "http://127.0.0.1:5000"
    assert calls["run_id"] == "existing-run-id"


def test_test_evaluation_logs_zenml_and_model_tags(monkeypatch):
    X_test, y_test = create_test_data()
    model = DummyModelWithProba()

    calls = patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        test_evaluation_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    _ = run_test_evaluation(
        model,
        X_test,
        y_test,
        model_name="random_forest",
    )

    assert calls["tags"]["zenml_run_id"] == "dummy-zenml-run-id"
    assert calls["tags"]["zenml_pipeline_run_name"] == "dummy-pipeline-run"
    assert calls["tags"]["evaluation_model_name"] == "random_forest"


def test_test_evaluation_logs_all_test_metrics(monkeypatch):
    X_test, y_test = create_test_data()
    model = DummyModelWithProba()

    calls = patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        test_evaluation_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    test_metrics, _, _ = run_test_evaluation(model, X_test, y_test)

    for metric_name, metric_value in test_metrics.items():
        logged_metric_name = f"test_{metric_name}"

        assert logged_metric_name in calls["metrics"]
        assert calls["metrics"][logged_metric_name] == metric_value


def test_test_evaluation_does_not_modify_input_data(monkeypatch):
    X_test, y_test = create_test_data()
    original_X_test = X_test.copy(deep=True)
    original_y_test = y_test.copy(deep=True)

    model = DummyModelWithProba()

    patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        test_evaluation_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    _ = run_test_evaluation(model, X_test, y_test)

    pd.testing.assert_frame_equal(X_test, original_X_test)
    pd.testing.assert_series_equal(y_test, original_y_test)