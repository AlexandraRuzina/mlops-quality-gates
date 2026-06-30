import sys
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CALCULATE_DUMMY_METRICS_PATH = (
    PROJECT_ROOT / "steps" / "finalCheck" / "calculate_dummy_metrics.py"
)

sys.path.append(str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "calculate_dummy_metrics",
    CALCULATE_DUMMY_METRICS_PATH,
)
calculate_dummy_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calculate_dummy_metrics)


def run_dummy_baseline(X_train, X_test, y_train, y_test):
    return calculate_dummy_metrics.evaluate_dummy_baseline.entrypoint(
        X_train,
        X_test,
        y_train,
        y_test,
    )


def test_evaluate_dummy_baseline_returns_float():
    X_train = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_train = pd.Series([0, 0, 0, 1])

    X_test = pd.DataFrame({"feature": [5, 6]})
    y_test = pd.Series([0, 1])

    result = run_dummy_baseline(X_train, X_test, y_train, y_test)

    assert isinstance(result, float)


def test_evaluate_dummy_baseline_uses_most_frequent_class():
    X_train = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
    y_train = pd.Series([0, 0, 0, 0, 1])

    X_test = pd.DataFrame({"feature": [6, 7, 8, 9]})
    y_test = pd.Series([0, 0, 1, 1])

    result = run_dummy_baseline(X_train, X_test, y_train, y_test)

    assert result == 0.5


def test_evaluate_dummy_baseline_returns_perfect_accuracy_when_test_contains_only_majority_class():
    X_train = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_train = pd.Series([1, 1, 1, 0])

    X_test = pd.DataFrame({"feature": [5, 6, 7]})
    y_test = pd.Series([1, 1, 1])

    result = run_dummy_baseline(X_train, X_test, y_train, y_test)

    assert result == 1.0


def test_evaluate_dummy_baseline_returns_zero_when_test_contains_only_non_majority_class():
    X_train = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_train = pd.Series([0, 0, 0, 1])

    X_test = pd.DataFrame({"feature": [5, 6, 7]})
    y_test = pd.Series([1, 1, 1])

    result = run_dummy_baseline(X_train, X_test, y_train, y_test)

    assert result == 0.0


def test_evaluate_dummy_baseline_ignores_feature_values():
    X_train = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_train = pd.Series([0, 0, 0, 1])

    X_test_a = pd.DataFrame({"feature": [5, 6, 7, 8]})
    X_test_b = pd.DataFrame({"feature": [999, -100, 42, 0]})
    y_test = pd.Series([0, 1, 0, 1])

    result_a = run_dummy_baseline(X_train, X_test_a, y_train, y_test)
    result_b = run_dummy_baseline(X_train, X_test_b, y_train, y_test)

    assert result_a == result_b
    assert result_a == 0.5


def test_evaluate_dummy_baseline_matches_expected_accuracy_for_credit_distribution():
    X_train = pd.DataFrame({"feature": range(10)})
    y_train = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

    X_test = pd.DataFrame({"feature": range(10, 20)})
    y_test = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    result = run_dummy_baseline(X_train, X_test, y_train, y_test)

    assert result == pytest.approx(0.6)