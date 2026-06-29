from zenml import step
from typing import Annotated

@step
def performance_gate(
    model_metrics: dict,
    dummy_accuracy: float,
) -> tuple[
    Annotated[dict, "performance_report"],
    Annotated[bool, "performance_gate_passed"],
]:
    """
    Performance Quality Gate.

    Evaluates whether the trained model satisfies the required
    performance thresholds and returns a structured gate report.
    """

    thresholds = {
        "accuracy": dummy_accuracy,
        "precision": 0.50,
        "recall": 0.70,
        "f1": 0.60,
        "roc_auc": 0.80,
        "mcc": 0.40,
    }

    metric_results = {
        "accuracy": {
            "value": model_metrics["accuracy"],
            "threshold": dummy_accuracy,
            "comparison": ">",
            "passed": model_metrics["accuracy"] > dummy_accuracy,
        },
        "precision": {
            "value": model_metrics["precision"],
            "threshold": thresholds["precision"],
            "comparison": ">",
            "passed": model_metrics["precision"] > thresholds["precision"],
        },
        "recall": {
            "value": model_metrics["recall"],
            "threshold": thresholds["recall"],
            "comparison": ">",
            "passed": model_metrics["recall"] > thresholds["recall"],
        },
        "f1": {
            "value": model_metrics["f1"],
            "threshold": thresholds["f1"],
            "comparison": ">",
            "passed": model_metrics["f1"] > thresholds["f1"],
        },
        "roc_auc": {
            "value": model_metrics["roc_auc"],
            "threshold": thresholds["roc_auc"],
            "comparison": ">",
            "passed": model_metrics["roc_auc"] > thresholds["roc_auc"],
        },
        "mcc": {
            "value": model_metrics["mcc"],
            "threshold": thresholds["mcc"],
            "comparison": ">",
            "passed": model_metrics["mcc"] > thresholds["mcc"],
        },
    }

    gate_passed = all(
        result["passed"] for result in metric_results.values()
    )

    performance_report = {
        "gate_name": "Performance Quality Gate",
        "gate_passed": gate_passed,
        "metrics": metric_results,
    }

    print("\n" + "=" * 60)
    print("Performance Quality Gate")
    print("=" * 60)

    for metric, result in metric_results.items():
        status = "PASSED" if result["passed"] else "FAILED"
        print(
            f"{metric:<12} "
            f"value={result['value']:.4f} "
            f"threshold {result['comparison']} {result['threshold']:.4f} "
            f"{status}"
        )

    print("-" * 60)
    print(f"Overall Gate Status: {'PASSED' if gate_passed else 'FAILED'}")
    print("=" * 60)

    return performance_report, gate_passed