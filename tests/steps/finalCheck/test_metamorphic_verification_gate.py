import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
METAMORPHIC_GATE_PATH = (
    PROJECT_ROOT / "steps" / "finalCheck" / "metamorphic_verification_gate.py"
)

sys.path.append(str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "metamorphic_verification_gate",
    METAMORPHIC_GATE_PATH,
)
metamorphic_gate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metamorphic_gate_module)


class DummyModel:
    def __init__(self, probability_outputs):
        self.probability_outputs = probability_outputs
        self.call_count = 0

    def predict_proba(self, X):
        probabilities_bad = self.probability_outputs[self.call_count]
        self.call_count += 1

        return np.column_stack(
            [
                1 - probabilities_bad,
                probabilities_bad,
            ]
        )


def create_X_test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "duration": [12, 24, 36, 48],
            "credit_amount": [1000, 2000, 3000, 4000],
            "monthly_payment": [83.33, 83.33, 83.33, 83.33],
            "checking_status": ["<0", "0<=X<200", ">=200", "no checking"],
            "savings_status": ["100<=X<500", "500<=X<1000", ">=1000", "<100"],
        }
    )


def patch_metamorphic_generators(monkeypatch):
    def fake_generate_duration_increase_data(X_test):
        transformed = X_test.copy()
        transformed["duration"] = transformed["duration"] + 12
        return transformed

    def fake_generate_checking_status_improvement_data(X_test):
        transformed = X_test.copy()
        mask = pd.Series(
            [True, True, False, False],
            index=X_test.index,
        )
        return transformed, mask

    def fake_generate_savings_status_deterioration_data(X_test):
        transformed = X_test.copy()
        mask = pd.Series(
            [True, True, True, False],
            index=X_test.index,
        )
        return transformed, mask

    monkeypatch.setattr(
        metamorphic_gate_module,
        "generate_duration_increase_data",
        fake_generate_duration_increase_data,
    )

    monkeypatch.setattr(
        metamorphic_gate_module,
        "generate_checking_status_improvement_data",
        fake_generate_checking_status_improvement_data,
    )

    monkeypatch.setattr(
        metamorphic_gate_module,
        "generate_savings_status_deterioration_data",
        fake_generate_savings_status_deterioration_data,
    )


def run_metamorphic_gate(model, X_test, y_proba):
    return metamorphic_gate_module.metamorphic_verification_gate.entrypoint(
        model,
        X_test,
        y_proba,
    )


def test_metamorphic_gate_returns_report_and_boolean(monkeypatch):
    X_test = create_X_test()
    y_proba = np.array([0.30, 0.40, 0.50, 0.60])

    patch_metamorphic_generators(monkeypatch)

    model = DummyModel(
        probability_outputs=[
            np.array([0.35, 0.45, 0.55, 0.65]),
            np.array([0.25, 0.35, 0.50, 0.60]),
            np.array([0.35, 0.45, 0.55, 0.60]),
        ]
    )

    report, gate_passed = run_metamorphic_gate(model, X_test, y_proba)

    assert isinstance(report, dict)
    assert isinstance(gate_passed, bool)


def test_metamorphic_gate_report_has_expected_structure(monkeypatch):
    X_test = create_X_test()
    y_proba = np.array([0.30, 0.40, 0.50, 0.60])

    patch_metamorphic_generators(monkeypatch)

    model = DummyModel(
        probability_outputs=[
            np.array([0.35, 0.45, 0.55, 0.65]),
            np.array([0.25, 0.35, 0.50, 0.60]),
            np.array([0.35, 0.45, 0.55, 0.60]),
        ]
    )

    report, _ = run_metamorphic_gate(model, X_test, y_proba)

    assert report["gate_name"] == "Metamorphic Verification Gate"
    assert report["max_violation_rate"] == 0.1
    assert "gate_passed" in report
    assert "metrics" in report


def test_metamorphic_gate_contains_all_relations(monkeypatch):
    X_test = create_X_test()
    y_proba = np.array([0.30, 0.40, 0.50, 0.60])

    patch_metamorphic_generators(monkeypatch)

    model = DummyModel(
        probability_outputs=[
            np.array([0.35, 0.45, 0.55, 0.65]),
            np.array([0.25, 0.35, 0.50, 0.60]),
            np.array([0.35, 0.45, 0.55, 0.60]),
        ]
    )

    report, _ = run_metamorphic_gate(model, X_test, y_proba)

    expected_relations = {
        "duration_increase",
        "checking_status_improvement",
        "savings_status_deterioration",
    }

    assert set(report["metrics"].keys()) == expected_relations


def test_metamorphic_gate_passes_when_no_relation_violates(monkeypatch):
    X_test = create_X_test()
    y_proba = np.array([0.30, 0.40, 0.50, 0.60])

    patch_metamorphic_generators(monkeypatch)

    model = DummyModel(
        probability_outputs=[
            np.array([0.30, 0.40, 0.50, 0.60]),
            np.array([0.30, 0.40, 0.50, 0.60]),
            np.array([0.30, 0.40, 0.50, 0.60]),
        ]
    )

    report, gate_passed = run_metamorphic_gate(model, X_test, y_proba)

    assert gate_passed is True
    assert report["gate_passed"] is True

    for relation_result in report["metrics"].values():
        assert bool(relation_result["passed"]) is True
        assert relation_result["number_of_violations"] == 0


def test_metamorphic_gate_fails_when_duration_relation_has_too_many_violations(monkeypatch):
    X_test = create_X_test()
    y_proba = np.array([0.30, 0.40, 0.50, 0.60])

    patch_metamorphic_generators(monkeypatch)

    model = DummyModel(
        probability_outputs=[
            np.array([0.20, 0.30, 0.40, 0.50]),
            np.array([0.30, 0.40, 0.50, 0.60]),
            np.array([0.30, 0.40, 0.50, 0.60]),
        ]
    )

    report, gate_passed = run_metamorphic_gate(model, X_test, y_proba)

    assert gate_passed is False
    assert report["gate_passed"] is False
    assert report["metrics"]["duration_increase"]["number_of_violations"] == 4
    assert report["metrics"]["duration_increase"]["violation_rate"] == 1.0
    assert bool(report["metrics"]["duration_increase"]["passed"]) is False


def test_metamorphic_gate_fails_when_checking_relation_has_too_many_violations(monkeypatch):
    X_test = create_X_test()
    y_proba = np.array([0.30, 0.40, 0.50, 0.60])

    patch_metamorphic_generators(monkeypatch)

    model = DummyModel(
        probability_outputs=[
            np.array([0.30, 0.40, 0.50, 0.60]),
            np.array([0.35, 0.45, 0.50, 0.60]),
            np.array([0.30, 0.40, 0.50, 0.60]),
        ]
    )

    report, gate_passed = run_metamorphic_gate(model, X_test, y_proba)

    assert gate_passed is False
    assert report["metrics"]["checking_status_improvement"]["number_of_cases"] == 2
    assert report["metrics"]["checking_status_improvement"]["number_of_violations"] == 2
    assert report["metrics"]["checking_status_improvement"]["violation_rate"] == 1.0


def test_metamorphic_gate_fails_when_savings_relation_has_too_many_violations(monkeypatch):
    X_test = create_X_test()
    y_proba = np.array([0.30, 0.40, 0.50, 0.60])

    patch_metamorphic_generators(monkeypatch)

    model = DummyModel(
        probability_outputs=[
            np.array([0.30, 0.40, 0.50, 0.60]),
            np.array([0.30, 0.40, 0.50, 0.60]),
            np.array([0.20, 0.30, 0.40, 0.60]),
        ]
    )

    report, gate_passed = run_metamorphic_gate(model, X_test, y_proba)

    assert gate_passed is False
    assert report["metrics"]["savings_status_deterioration"]["number_of_cases"] == 3
    assert report["metrics"]["savings_status_deterioration"]["number_of_violations"] == 3
    assert report["metrics"]["savings_status_deterioration"]["violation_rate"] == 1.0


def test_metamorphic_gate_uses_threshold_and_comparison_for_each_relation(monkeypatch):
    X_test = create_X_test()
    y_proba = np.array([0.30, 0.40, 0.50, 0.60])

    patch_metamorphic_generators(monkeypatch)

    model = DummyModel(
        probability_outputs=[
            np.array([0.30, 0.40, 0.50, 0.60]),
            np.array([0.30, 0.40, 0.50, 0.60]),
            np.array([0.30, 0.40, 0.50, 0.60]),
        ]
    )

    report, _ = run_metamorphic_gate(model, X_test, y_proba)

    for relation_result in report["metrics"].values():
        assert relation_result["threshold"] == 0.1
        assert relation_result["comparison"] == "<="


def test_metamorphic_gate_calls_predict_proba_three_times(monkeypatch):
    X_test = create_X_test()
    y_proba = np.array([0.30, 0.40, 0.50, 0.60])

    patch_metamorphic_generators(monkeypatch)

    model = DummyModel(
        probability_outputs=[
            np.array([0.30, 0.40, 0.50, 0.60]),
            np.array([0.30, 0.40, 0.50, 0.60]),
            np.array([0.30, 0.40, 0.50, 0.60]),
        ]
    )

    _ = run_metamorphic_gate(model, X_test, y_proba)

    assert model.call_count == 3


def test_metamorphic_gate_does_not_modify_original_x_test(monkeypatch):
    X_test = create_X_test()
    original_X_test = X_test.copy(deep=True)
    y_proba = np.array([0.30, 0.40, 0.50, 0.60])

    patch_metamorphic_generators(monkeypatch)

    model = DummyModel(
        probability_outputs=[
            np.array([0.30, 0.40, 0.50, 0.60]),
            np.array([0.30, 0.40, 0.50, 0.60]),
            np.array([0.30, 0.40, 0.50, 0.60]),
        ]
    )

    _ = run_metamorphic_gate(model, X_test, y_proba)

    pd.testing.assert_frame_equal(X_test, original_X_test)