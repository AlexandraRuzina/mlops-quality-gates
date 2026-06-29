from zenml import step
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import Annotated
import numpy as np


@step
def fairness_gate(
    y_test: Annotated[pd.Series, "True labels"],
    y_pred: Annotated[np.ndarray, "Predicted labels"],
    sex_test: Annotated[pd.Series, "Sensitive attribute"],
) -> tuple[
    Annotated[dict, "fairness_report"],
    Annotated[bool, "fairness_gate_passed"],
]:
    """
    Analyzes fairness based on the separation criterion.

    Separation requires similar error rates across sensitive groups.
    Here, we compare False Positive Rate (FPR) and False Negative Rate (FNR)
    between male and female applicants.
    """

    threshold = 0.05

    fairness_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "sex": sex_test,
    })

    group_results = {}

    print("\n" + "=" * 60)
    print("Fairness Quality Gate")
    print("=" * 60)

    for group in fairness_df["sex"].unique():

        group_df = fairness_df[fairness_df["sex"] == group]

        tn, fp, fn, tp = confusion_matrix(
            group_df["y_true"],
            group_df["y_pred"],
            labels=[0, 1],
        ).ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

        group_results[group] = {
            "samples": len(group_df),
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "true_positive_rate": tpr,
        }

        print(f"\nGroup: {group}")
        print(f"Samples               : {len(group_df)}")
        print(f"False Positive Rate   : {fpr:.4f}")
        print(f"False Negative Rate   : {fnr:.4f}")
        print(f"True Positive Rate    : {tpr:.4f}")

    if "male" not in group_results or "female" not in group_results:
        raise ValueError(
            "Fairness Analysis failed: both groups 'male' and 'female' must be present."
        )

    fpr_difference = abs(
        group_results["male"]["false_positive_rate"]
        - group_results["female"]["false_positive_rate"]
    )

    fnr_difference = abs(
        group_results["male"]["false_negative_rate"]
        - group_results["female"]["false_negative_rate"]
    )

    separation_difference = (fpr_difference + fnr_difference) / 2

    metric_results = {
        "separation_difference": {
            "value": separation_difference,
            "threshold": threshold,
            "comparison": "<=",
            "passed": separation_difference <= threshold,
        }
    }

    gate_passed = all(
        result["passed"] for result in metric_results.values()
    )

    fairness_report = {
        "gate_name": "Fairness Quality Gate",
        "gate_passed": gate_passed,
        "groups": group_results,
        "metrics": {
            "fpr_difference": fpr_difference,
            "fnr_difference": fnr_difference,
            **metric_results,
        },
    }

    print("\n" + "-" * 60)
    print("Fairness Evaluation")
    print("-" * 60)
    print(f"FPR Difference         : {fpr_difference:.4f}")
    print(f"FNR Difference         : {fnr_difference:.4f}")

    status = (
        "PASSED"
        if metric_results["separation_difference"]["passed"]
        else "FAILED"
    )

    print(
        f"Separation Difference  : "
        f"{separation_difference:.4f} "
        f"(threshold <= {threshold:.4f}) "
        f"{status}"
    )

    print("-" * 60)
    print(f"Overall Gate Status: {'PASSED' if gate_passed else 'FAILED'}")
    print("=" * 60)

    return fairness_report, gate_passed