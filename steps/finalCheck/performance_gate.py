from zenml import step


@step
def performance_gate(
    model_metrics: dict,
    dummy_accuracy: float,
) -> None:
    """
    Performance Quality Gate.

    Validates whether the trained model satisfies the required
    performance thresholds before deployment.
    """

    print("\n" + "=" * 60)
    print("Performance Quality Gate")
    print("=" * 60)

    print(f"Dummy Accuracy : {dummy_accuracy:.4f}")
    print(f"Accuracy       : {model_metrics['accuracy']:.4f}")
    print(f"Precision      : {model_metrics['precision']:.4f}")
    print(f"Recall         : {model_metrics['recall']:.4f}")
    print(f"F1-Score       : {model_metrics['f1']:.4f}")
    print(f"ROC-AUC        : {model_metrics['roc_auc']:.4f}")
    print(f"MCC            : {model_metrics['mcc']:.4f}")

    checks = {
        "Accuracy > Dummy": model_metrics["accuracy"] > dummy_accuracy,
        "Precision > 0.50": model_metrics["precision"] > 0.50,
        "Recall > 0.70": model_metrics["recall"] > 0.70,
        "F1-Score > 0.60": model_metrics["f1"] > 0.60,
        "ROC-AUC > 0.80": model_metrics["roc_auc"] > 0.80,
        "MCC > 0.40": model_metrics["mcc"] > 0.40,
    }

    print("\nGate Results")
    print("-" * 60)

    for check, passed in checks.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{check:<25} {status}")

    if not all(checks.values()):
        failed_checks = [
            check for check, passed in checks.items() if not passed
        ]

        raise ValueError(
            "Performance Quality Gate failed.\n"
            f"Failed checks: {', '.join(failed_checks)}"
        )

    print("\nPerformance Quality Gate passed successfully.")
    print("=" * 60)