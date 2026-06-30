import sys
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PERFORMANCE_GATE_PATH = (
    PROJECT_ROOT / "steps" / "finalCheck" / "performance_gate.py"
)

sys.path.append(str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "performance_gate",
    PERFORMANCE_GATE_PATH,
)
performance_gate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(performance_gate_module)


def run_performance_gate(model_metrics: dict, dummy_accuracy: float):
    return performance_gate_module.performance_gate.entrypoint(
        model_metrics,
        dummy_accuracy,
    )


def create_passing_metrics():
    return {
        "accuracy": 0.75,
        "precision": 0.60,
        "recall": 0.75,
        "f1": 0.65,
        "roc_auc": 0.85,
        "mcc": 0.45,
    }


def test_performance_gate_returns_report_and_boolean():
    model_metrics = create_passing_metrics()

    performance_report, gate_passed = run_performance_gate(
        model_metrics=model_metrics,
        dummy_accuracy=0.70,
    )

    assert isinstance(performance_report, dict)
    assert isinstance(gate_passed, bool)


def test_performance_gate_passes_when_all_metrics_exceed_thresholds():
    model_metrics = create_passing_metrics()

    performance_report, gate_passed = run_performance_gate(
        model_metrics=model_metrics,
        dummy_accuracy=0.70,
    )

    assert gate_passed is True
    assert performance_report["gate_passed"] is True


def test_performance_gate_fails_when_accuracy_not_above_dummy():
    model_metrics = create_passing_metrics()
    model_metrics["accuracy"] = 0.69

    performance_report, gate_passed = run_performance_gate(
        model_metrics=model_metrics,
        dummy_accuracy=0.70,
    )

    assert gate_passed is False
    assert performance_report["metrics"]["accuracy"]["passed"] is False


def test_performance_gate_fails_when_precision_below_threshold():
    model_metrics = create_passing_metrics()
    model_metrics["precision"] = 0.49

    performance_report, gate_passed = run_performance_gate(
        model_metrics=model_metrics,
        dummy_accuracy=0.70,
    )

    assert gate_passed is False
    assert performance_report["metrics"]["precision"]["passed"] is False


def test_performance_gate_fails_when_recall_below_threshold():
    model_metrics = create_passing_metrics()
    model_metrics["recall"] = 0.69

    performance_report, gate_passed = run_performance_gate(
        model_metrics=model_metrics,
        dummy_accuracy=0.70,
    )

    assert gate_passed is False
    assert performance_report["metrics"]["recall"]["passed"] is False


def test_performance_gate_fails_when_f1_below_threshold():
    model_metrics = create_passing_metrics()
    model_metrics["f1"] = 0.59

    performance_report, gate_passed = run_performance_gate(
        model_metrics=model_metrics,
        dummy_accuracy=0.70,
    )

    assert gate_passed is False
    assert performance_report["metrics"]["f1"]["passed"] is False


def test_performance_gate_fails_when_roc_auc_below_threshold():
    model_metrics = create_passing_metrics()
    model_metrics["roc_auc"] = 0.79

    performance_report, gate_passed = run_performance_gate(
        model_metrics=model_metrics,
        dummy_accuracy=0.70,
    )

    assert gate_passed is False
    assert performance_report["metrics"]["roc_auc"]["passed"] is False


def test_performance_gate_fails_when_mcc_below_threshold():
    model_metrics = create_passing_metrics()
    model_metrics["mcc"] = 0.39

    performance_report, gate_passed = run_performance_gate(
        model_metrics=model_metrics,
        dummy_accuracy=0.70,
    )

    assert gate_passed is False
    assert performance_report["metrics"]["mcc"]["passed"] is False


def test_performance_gate_uses_strict_greater_than_thresholds():
    model_metrics = {
        "accuracy": 0.70,
        "precision": 0.50,
        "recall": 0.70,
        "f1": 0.60,
        "roc_auc": 0.80,
        "mcc": 0.40,
    }

    performance_report, gate_passed = run_performance_gate(
        model_metrics=model_metrics,
        dummy_accuracy=0.70,
    )

    assert gate_passed is False

    for metric_result in performance_report["metrics"].values():
        assert metric_result["passed"] is False


def test_performance_report_has_expected_structure():
    model_metrics = create_passing_metrics()

    performance_report, _ = run_performance_gate(
        model_metrics=model_metrics,
        dummy_accuracy=0.70,
    )

    assert performance_report["gate_name"] == "Performance Quality Gate"
    assert "gate_passed" in performance_report
    assert "metrics" in performance_report


def test_performance_report_contains_all_expected_metrics():
    model_metrics = create_passing_metrics()

    performance_report, _ = run_performance_gate(
        model_metrics=model_metrics,
        dummy_accuracy=0.70,
    )

    expected_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "mcc",
    }

    assert set(performance_report["metrics"].keys()) == expected_metrics


def test_performance_report_contains_values_thresholds_comparisons_and_status():
    model_metrics = create_passing_metrics()

    performance_report, _ = run_performance_gate(
        model_metrics=model_metrics,
        dummy_accuracy=0.70,
    )

    for metric_name, metric_result in performance_report["metrics"].items():
        assert "value" in metric_result
        assert "threshold" in metric_result
        assert "comparison" in metric_result
        assert "passed" in metric_result

        assert metric_result["value"] == model_metrics[metric_name]
        assert metric_result["comparison"] == ">"


def test_accuracy_threshold_is_dummy_accuracy():
    model_metrics = create_passing_metrics()

    performance_report, _ = run_performance_gate(
        model_metrics=model_metrics,
        dummy_accuracy=0.68,
    )

    assert performance_report["metrics"]["accuracy"]["threshold"] == 0.68


def test_fixed_metric_thresholds_are_correct():
    model_metrics = create_passing_metrics()

    performance_report, _ = run_performance_gate(
        model_metrics=model_metrics,
        dummy_accuracy=0.70,
    )

    assert performance_report["metrics"]["precision"]["threshold"] == 0.50
    assert performance_report["metrics"]["recall"]["threshold"] == 0.70
    assert performance_report["metrics"]["f1"]["threshold"] == 0.60
    assert performance_report["metrics"]["roc_auc"]["threshold"] == 0.80
    assert performance_report["metrics"]["mcc"]["threshold"] == 0.40


def test_performance_gate_fails_when_multiple_metrics_fail():
    model_metrics = create_passing_metrics()
    model_metrics["precision"] = 0.40
    model_metrics["mcc"] = 0.20

    performance_report, gate_passed = run_performance_gate(
        model_metrics=model_metrics,
        dummy_accuracy=0.70,
    )

    assert gate_passed is False
    assert performance_report["gate_passed"] is False
    assert performance_report["metrics"]["precision"]["passed"] is False
    assert performance_report["metrics"]["mcc"]["passed"] is False
    assert performance_report["metrics"]["accuracy"]["passed"] is True