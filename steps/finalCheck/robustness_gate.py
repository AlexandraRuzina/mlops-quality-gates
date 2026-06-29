from scripts.generate_robustness_test_data import (
    generate_perturbation,
    generate_inflation_test_data,
)
from typing import Any, Annotated

import pandas as pd
from sklearn.metrics import f1_score
from zenml import step


@step
def robustness_gate(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_metrics: dict,
) -> tuple[
    Annotated[dict, "robustness_report"],
    Annotated[bool, "robustness_gate_passed"],
]:
    """
    Evaluates robustness using perturbation and inflation test data.
    """

    threshold = 0.95

    f1_clean = test_metrics["f1"]

    perturbation_data = generate_perturbation(X_test)
    inflation_data = generate_inflation_test_data(X_test)

    y_pred_perturbation = model.predict(perturbation_data)
    y_pred_inflation = model.predict(inflation_data)

    f1_perturbation = f1_score(
        y_test,
        y_pred_perturbation,
        pos_label=1,
    )

    f1_inflation = f1_score(
        y_test,
        y_pred_inflation,
        pos_label=1,
    )

    robustness_perturbation = f1_perturbation / f1_clean
    robustness_inflation = f1_inflation / f1_clean

    mean_robustness_score = (
        robustness_perturbation + robustness_inflation
    ) / 2

    metric_results = {
        "mean_robustness_score": {
            "value": mean_robustness_score,
            "threshold": threshold,
            "comparison": ">=",
            "passed": mean_robustness_score >= threshold,
        }
    }

    gate_passed = all(
        result["passed"] for result in metric_results.values()
    )

    robustness_report = {
        "gate_name": "Robustness Quality Gate",
        "gate_passed": gate_passed,
        "metrics": {
            "f1_clean": f1_clean,
            "f1_perturbation": f1_perturbation,
            "f1_inflation": f1_inflation,
            "robustness_perturbation": robustness_perturbation,
            "robustness_inflation": robustness_inflation,
            **metric_results,
        },
    }

    print("\n" + "=" * 60)
    print("Robustness Quality Gate")
    print("=" * 60)

    print(f"F1 Clean              : {f1_clean:.4f}")
    print(f"F1 Perturbation       : {f1_perturbation:.4f}")
    print(f"F1 Inflation          : {f1_inflation:.4f}")
    print(f"Robustness Perturb.   : {robustness_perturbation:.4f}")
    print(f"Robustness Inflation  : {robustness_inflation:.4f}")

    status = (
        "PASSED"
        if metric_results["mean_robustness_score"]["passed"]
        else "FAILED"
    )

    print("-" * 60)
    print(
        f"Mean Robustness Score : "
        f"{mean_robustness_score:.4f} "
        f"(threshold >= {threshold:.4f}) "
        f"{status}"
    )

    print("-" * 60)
    print(f"Overall Gate Status: {'PASSED' if gate_passed else 'FAILED'}")
    print("=" * 60)

    return robustness_report, gate_passed