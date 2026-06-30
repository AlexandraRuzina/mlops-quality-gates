import sys
import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_READINESS_GATE_PATH = (
    PROJECT_ROOT / "steps" / "finalCheck" / "production_readiness_gate.py"
)

sys.path.append(str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "production_readiness_gate",
    PRODUCTION_READINESS_GATE_PATH,
)
production_readiness_gate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production_readiness_gate_module)


def run_production_readiness_gate(production_readiness_passed: bool):
    return production_readiness_gate_module.production_readiness_gate.entrypoint(
        production_readiness_passed,
    )


def test_production_readiness_gate_passes_when_all_gates_passed():
    result = run_production_readiness_gate(True)

    assert result is True


def test_production_readiness_gate_raises_error_when_any_gate_failed():
    with pytest.raises(
        ValueError,
        match="Production Readiness Gate failed",
    ):
        run_production_readiness_gate(False)


def test_production_readiness_gate_error_message_explains_model_not_registered():
    with pytest.raises(ValueError) as error:
        run_production_readiness_gate(False)

    assert "At least one quality gate did not pass" in str(error.value)
    assert "The model will not be registered" in str(error.value)