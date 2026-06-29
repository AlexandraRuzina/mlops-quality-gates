import sys
import importlib.util
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINAL_TRAINING_PATH = (
    PROJECT_ROOT / "steps" / "finalCheck" / "final_training_random_forest.py"
)

sys.path.append(str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "final_training_random_forest",
    FINAL_TRAINING_PATH,
)
final_training_random_forest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(final_training_random_forest)


class DummyPipelineRun:
    id = "dummy-zenml-run-id"
    name = "dummy-pipeline-run"


class DummyContext:
    pipeline_run = DummyPipelineRun()


class DummyMLflowRunInfo:
    run_id = "dummy-mlflow-run-id"


class DummyMLflowRun:
    info = DummyMLflowRunInfo()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class DummyPreprocessor:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X


class DummyRandomForestClassifier:
    last_instance = None

    def __init__(self, **kwargs):
        self.params = kwargs
        DummyRandomForestClassifier.last_instance = self

    def fit(self, X, y):
        self.X_fit_ = X
        self.y_fit_ = y
        self.fit_called_ = True
        return self


def create_sample_training_data():
    X_train = pd.DataFrame(
        [
            {
                "duration": 24,
                "credit_amount": 3000,
                "employment": "1<=X<4",
                "job": "skilled",
            },
            {
                "duration": 36,
                "credit_amount": 6000,
                "employment": ">=7",
                "job": "high qualif/self emp/mgmt",
            },
        ]
    )

    y_train = pd.Series([0, 1])

    return X_train, y_train


def patch_mlflow(monkeypatch):
    calls = {
        "tracking_uri": None,
        "experiment": None,
        "tags": {},
        "params": None,
        "logged_model": None,
        "run_name": None,
    }

    monkeypatch.setattr(
        final_training_random_forest.mlflow,
        "set_tracking_uri",
        lambda uri: calls.update({"tracking_uri": uri}),
    )

    monkeypatch.setattr(
        final_training_random_forest.mlflow,
        "set_experiment",
        lambda experiment: calls.update({"experiment": experiment}),
    )

    def fake_start_run(run_name):
        calls["run_name"] = run_name
        return DummyMLflowRun()

    monkeypatch.setattr(
        final_training_random_forest.mlflow,
        "start_run",
        fake_start_run,
    )

    monkeypatch.setattr(
        final_training_random_forest.mlflow,
        "set_tag",
        lambda key, value: calls["tags"].update({key: value}),
    )

    monkeypatch.setattr(
        final_training_random_forest.mlflow,
        "log_params",
        lambda params: calls.update({"params": params}),
    )

    monkeypatch.setattr(
        final_training_random_forest.mlflow.sklearn,
        "log_model",
        lambda sk_model, artifact_path: calls.update(
            {
                "logged_model": {
                    "sk_model": sk_model,
                    "artifact_path": artifact_path,
                }
            }
        ),
    )

    return calls


def run_final_training(X_train, y_train, best_params):
    return final_training_random_forest.train_final_random_forest.entrypoint(
        X_train,
        y_train,
        best_params,
    )


def test_train_final_random_forest_returns_pipeline_and_mlflow_run_id(monkeypatch):
    X_train, y_train = create_sample_training_data()
    best_params = {
        "n_estimators": 100,
        "max_depth": 5,
    }

    patch_mlflow(monkeypatch)

    monkeypatch.setattr(
        final_training_random_forest,
        "get_step_context",
        lambda: DummyContext(),
    )

    monkeypatch.setattr(
        final_training_random_forest,
        "create_random_forest_preprocessor",
        lambda X: DummyPreprocessor(),
    )

    monkeypatch.setattr(
        final_training_random_forest,
        "RandomForestClassifier",
        DummyRandomForestClassifier,
    )

    model_pipeline, mlflow_run_id = run_final_training(
        X_train,
        y_train,
        best_params,
    )

    assert isinstance(model_pipeline, Pipeline)
    assert mlflow_run_id == "dummy-mlflow-run-id"


def test_train_final_random_forest_configures_mlflow(monkeypatch):
    X_train, y_train = create_sample_training_data()
    best_params = {"n_estimators": 100}

    calls = patch_mlflow(monkeypatch)

    monkeypatch.setattr(
        final_training_random_forest,
        "get_step_context",
        lambda: DummyContext(),
    )
    monkeypatch.setattr(
        final_training_random_forest,
        "create_random_forest_preprocessor",
        lambda X: DummyPreprocessor(),
    )
    monkeypatch.setattr(
        final_training_random_forest,
        "RandomForestClassifier",
        DummyRandomForestClassifier,
    )

    _ = run_final_training(X_train, y_train, best_params)

    assert calls["tracking_uri"] == "http://127.0.0.1:5000"
    assert calls["experiment"] == "final_training"
    assert calls["run_name"] == "final_random_forest_training"


def test_train_final_random_forest_logs_zenml_tags(monkeypatch):
    X_train, y_train = create_sample_training_data()
    best_params = {"n_estimators": 100}

    calls = patch_mlflow(monkeypatch)

    monkeypatch.setattr(
        final_training_random_forest,
        "get_step_context",
        lambda: DummyContext(),
    )
    monkeypatch.setattr(
        final_training_random_forest,
        "create_random_forest_preprocessor",
        lambda X: DummyPreprocessor(),
    )
    monkeypatch.setattr(
        final_training_random_forest,
        "RandomForestClassifier",
        DummyRandomForestClassifier,
    )

    _ = run_final_training(X_train, y_train, best_params)

    assert calls["tags"]["zenml_run_id"] == "dummy-zenml-run-id"
    assert calls["tags"]["zenml_pipeline_run_name"] == "dummy-pipeline-run"


def test_train_final_random_forest_uses_best_params_and_fixed_params(monkeypatch):
    X_train, y_train = create_sample_training_data()
    best_params = {
        "n_estimators": 300,
        "max_depth": 5,
        "min_samples_split": 2,
    }

    patch_mlflow(monkeypatch)

    monkeypatch.setattr(
        final_training_random_forest,
        "get_step_context",
        lambda: DummyContext(),
    )
    monkeypatch.setattr(
        final_training_random_forest,
        "create_random_forest_preprocessor",
        lambda X: DummyPreprocessor(),
    )
    monkeypatch.setattr(
        final_training_random_forest,
        "RandomForestClassifier",
        DummyRandomForestClassifier,
    )

    _ = run_final_training(X_train, y_train, best_params)

    model = DummyRandomForestClassifier.last_instance

    assert model.params["n_estimators"] == 300
    assert model.params["max_depth"] == 5
    assert model.params["min_samples_split"] == 2
    assert model.params["random_state"] == 42
    assert model.params["class_weight"] == "balanced"
    assert model.params["n_jobs"] == 1


def test_train_final_random_forest_fits_pipeline(monkeypatch):
    X_train, y_train = create_sample_training_data()
    best_params = {"n_estimators": 100}

    patch_mlflow(monkeypatch)

    monkeypatch.setattr(
        final_training_random_forest,
        "get_step_context",
        lambda: DummyContext(),
    )
    monkeypatch.setattr(
        final_training_random_forest,
        "create_random_forest_preprocessor",
        lambda X: DummyPreprocessor(),
    )
    monkeypatch.setattr(
        final_training_random_forest,
        "RandomForestClassifier",
        DummyRandomForestClassifier,
    )

    _ = run_final_training(X_train, y_train, best_params)

    model = DummyRandomForestClassifier.last_instance

    assert model.fit_called_ is True
    pd.testing.assert_frame_equal(model.X_fit_, X_train)
    pd.testing.assert_series_equal(model.y_fit_, y_train)


def test_train_final_random_forest_logs_best_params(monkeypatch):
    X_train, y_train = create_sample_training_data()
    best_params = {
        "n_estimators": 300,
        "max_depth": 5,
    }

    calls = patch_mlflow(monkeypatch)

    monkeypatch.setattr(
        final_training_random_forest,
        "get_step_context",
        lambda: DummyContext(),
    )
    monkeypatch.setattr(
        final_training_random_forest,
        "create_random_forest_preprocessor",
        lambda X: DummyPreprocessor(),
    )
    monkeypatch.setattr(
        final_training_random_forest,
        "RandomForestClassifier",
        DummyRandomForestClassifier,
    )

    _ = run_final_training(X_train, y_train, best_params)

    assert calls["params"] == best_params


def test_train_final_random_forest_logs_model(monkeypatch):
    X_train, y_train = create_sample_training_data()
    best_params = {"n_estimators": 100}

    calls = patch_mlflow(monkeypatch)

    monkeypatch.setattr(
        final_training_random_forest,
        "get_step_context",
        lambda: DummyContext(),
    )
    monkeypatch.setattr(
        final_training_random_forest,
        "create_random_forest_preprocessor",
        lambda X: DummyPreprocessor(),
    )
    monkeypatch.setattr(
        final_training_random_forest,
        "RandomForestClassifier",
        DummyRandomForestClassifier,
    )

    model_pipeline, _ = run_final_training(X_train, y_train, best_params)

    assert calls["logged_model"]["artifact_path"] == "final_random_forest_model"
    assert calls["logged_model"]["sk_model"] is model_pipeline