import sys
import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTER_MODEL_PATH = (
    PROJECT_ROOT / "steps" / "register_model.py"
)

sys.path.append(str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "register_model",
    REGISTER_MODEL_PATH,
)
register_model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(register_model_module)


class DummyRegisteredModel:
    version = 7


def patch_mlflow(monkeypatch):
    calls = {
        "tracking_uri": None,
        "model_uri": None,
        "model_name": None,
    }

    monkeypatch.setattr(
        register_model_module.mlflow,
        "set_tracking_uri",
        lambda uri: calls.update({"tracking_uri": uri}),
    )

    def fake_register_model(model_uri, name):
        calls["model_uri"] = model_uri
        calls["model_name"] = name
        return DummyRegisteredModel()

    monkeypatch.setattr(
        register_model_module.mlflow,
        "register_model",
        fake_register_model,
    )

    return calls


def run_register_model(
    checked=True,
    mlflow_run_id="dummy-run-id",
    model_name="credit_risk_random_forest",
):
    return register_model_module.register_model.entrypoint(
        checked,
        mlflow_run_id,
        model_name,
    )


def test_register_model_returns_dictionary(monkeypatch):
    patch_mlflow(monkeypatch)

    result = run_register_model()

    assert isinstance(result, dict)


def test_register_model_returns_expected_information(monkeypatch):
    patch_mlflow(monkeypatch)

    result = run_register_model(
        checked=True,
        mlflow_run_id="run-123",
        model_name="credit_model",
    )

    assert result == {
        "registered_model_name": "credit_model",
        "model_uri": "runs:/run-123/final_random_forest_model",
        "run_id": "run-123",
        "model_version": 7,
    }


def test_register_model_sets_tracking_uri(monkeypatch):
    calls = patch_mlflow(monkeypatch)

    run_register_model()

    assert calls["tracking_uri"] == "http://127.0.0.1:5000"


def test_register_model_registers_correct_model(monkeypatch):
    calls = patch_mlflow(monkeypatch)

    run_register_model(
        checked=True,
        mlflow_run_id="abc123",
        model_name="my_credit_model",
    )

    assert calls["model_uri"] == "runs:/abc123/final_random_forest_model"
    assert calls["model_name"] == "my_credit_model"


def test_register_model_raises_error_when_checks_failed(monkeypatch):
    patch_mlflow(monkeypatch)

    with pytest.raises(
        ValueError,
        match="Model registration failed",
    ):
        run_register_model(
            checked=False,
        )


def test_register_model_does_not_register_when_checks_failed(monkeypatch):
    register_called = False

    monkeypatch.setattr(
        register_model_module.mlflow,
        "set_tracking_uri",
        lambda uri: None,
    )

    def fake_register_model(model_uri, name):
        nonlocal register_called
        register_called = True
        return DummyRegisteredModel()

    monkeypatch.setattr(
        register_model_module.mlflow,
        "register_model",
        fake_register_model,
    )

    with pytest.raises(ValueError):
        run_register_model(
            checked=False,
        )

    assert register_called is False


def test_register_model_uses_default_model_name(monkeypatch):
    calls = patch_mlflow(monkeypatch)

    run_register_model(
        checked=True,
        mlflow_run_id="run-42",
    )

    assert calls["model_name"] == "credit_risk_random_forest"


def test_register_model_builds_correct_model_uri(monkeypatch):
    patch_mlflow(monkeypatch)

    result = run_register_model(
        checked=True,
        mlflow_run_id="my-run-id",
    )

    assert result["model_uri"] == (
        "runs:/my-run-id/final_random_forest_model"
    )