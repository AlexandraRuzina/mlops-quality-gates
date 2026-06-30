import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FAIRNESS_GATE_PATH = (
    PROJECT_ROOT / "steps" / "finalCheck" / "fairness_gate.py"
)

sys.path.append(str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "fairness_gate",
    FAIRNESS_GATE_PATH,
)
fairness_gate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fairness_gate_module)


def run_fairness_gate(y_test, y_pred, sex_test):
    return fairness_gate_module.fairness_gate.entrypoint(
        y_test,
        y_pred,
        sex_test,
    )


def test_fairness_gate_returns_report_and_boolean():
    y_test = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    sex_test = pd.Series(
        ["male", "male", "male", "male", "female", "female", "female", "female"]
    )

    fairness_report, gate_passed = run_fairness_gate(
        y_test,
        y_pred,
        sex_test,
    )

    assert isinstance(fairness_report, dict)
    assert isinstance(gate_passed, bool)


def test_fairness_gate_report_has_expected_structure():
    y_test = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    sex_test = pd.Series(
        ["male", "male", "male", "male", "female", "female", "female", "female"]
    )

    fairness_report, _ = run_fairness_gate(
        y_test,
        y_pred,
        sex_test,
    )

    assert fairness_report["gate_name"] == "Fairness Quality Gate"
    assert "gate_passed" in fairness_report
    assert "groups" in fairness_report
    assert "metrics" in fairness_report


def test_fairness_gate_calculates_group_metrics_correctly():
    y_test = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1])
    sex_test = pd.Series(
        ["male", "male", "male", "male", "female", "female", "female", "female"]
    )

    fairness_report, _ = run_fairness_gate(
        y_test,
        y_pred,
        sex_test,
    )

    male = fairness_report["groups"]["male"]
    female = fairness_report["groups"]["female"]

    assert male["samples"] == 4
    assert male["false_positive_rate"] == pytest.approx(0.5)
    assert male["false_negative_rate"] == pytest.approx(0.5)
    assert male["true_positive_rate"] == pytest.approx(0.5)

    assert female["samples"] == 4
    assert female["false_positive_rate"] == pytest.approx(0.0)
    assert female["false_negative_rate"] == pytest.approx(0.0)
    assert female["true_positive_rate"] == pytest.approx(1.0)


def test_fairness_gate_calculates_differences_correctly():
    y_test = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1])
    sex_test = pd.Series(
        ["male", "male", "male", "male", "female", "female", "female", "female"]
    )

    fairness_report, _ = run_fairness_gate(
        y_test,
        y_pred,
        sex_test,
    )

    metrics = fairness_report["metrics"]

    assert metrics["fpr_difference"] == pytest.approx(0.5)
    assert metrics["fnr_difference"] == pytest.approx(0.5)
    assert metrics["separation_difference"]["value"] == pytest.approx(0.5)


def test_fairness_gate_passes_when_separation_difference_is_below_threshold():
    y_test = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    sex_test = pd.Series(
        ["male", "male", "male", "male", "female", "female", "female", "female"]
    )

    fairness_report, gate_passed = run_fairness_gate(
        y_test,
        y_pred,
        sex_test,
    )

    assert gate_passed is True
    assert fairness_report["gate_passed"] is True
    assert bool(fairness_report["metrics"]["separation_difference"]["passed"]) is True


def test_fairness_gate_fails_when_separation_difference_exceeds_threshold():
    y_test = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1])
    sex_test = pd.Series(
        ["male", "male", "male", "male", "female", "female", "female", "female"]
    )

    fairness_report, gate_passed = run_fairness_gate(
        y_test,
        y_pred,
        sex_test,
    )

    assert gate_passed is False
    assert fairness_report["gate_passed"] is False
    assert bool(fairness_report["metrics"]["separation_difference"]["passed"]) is False


def test_fairness_gate_uses_threshold_of_five_percentage_points():
    y_test = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    sex_test = pd.Series(
        ["male", "male", "male", "male", "female", "female", "female", "female"]
    )

    fairness_report, _ = run_fairness_gate(
        y_test,
        y_pred,
        sex_test,
    )

    assert (
        fairness_report["metrics"]["separation_difference"]["threshold"]
        == 0.05
    )
    assert (
        fairness_report["metrics"]["separation_difference"]["comparison"]
        == "<="
    )


def test_fairness_gate_raises_error_when_female_group_missing():
    y_test = pd.Series([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    sex_test = pd.Series(["male", "male", "male", "male"])

    with pytest.raises(
        ValueError,
        match="both groups 'male' and 'female' must be present",
    ):
        run_fairness_gate(y_test, y_pred, sex_test)


def test_fairness_gate_raises_error_when_male_group_missing():
    y_test = pd.Series([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    sex_test = pd.Series(["female", "female", "female", "female"])

    with pytest.raises(
        ValueError,
        match="both groups 'male' and 'female' must be present",
    ):
        run_fairness_gate(y_test, y_pred, sex_test)


def test_fairness_gate_handles_group_without_negative_cases():
    y_test = pd.Series([1, 1, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    sex_test = pd.Series(
        ["male", "male", "male", "male", "female", "female", "female", "female"]
    )

    fairness_report, _ = run_fairness_gate(
        y_test,
        y_pred,
        sex_test,
    )

    assert fairness_report["groups"]["male"]["false_positive_rate"] == 0


def test_fairness_gate_handles_group_without_positive_cases():
    y_test = pd.Series([0, 0, 0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    sex_test = pd.Series(
        ["male", "male", "male", "male", "female", "female", "female", "female"]
    )

    fairness_report, _ = run_fairness_gate(
        y_test,
        y_pred,
        sex_test,
    )

    assert fairness_report["groups"]["male"]["false_negative_rate"] == 0
    assert fairness_report["groups"]["male"]["true_positive_rate"] == 0