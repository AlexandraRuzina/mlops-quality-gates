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
) -> dict:
    """
    Analyzes fairness based on the separation criterion.

    Separation requires similar error rates across sensitive groups.
    Here, we compare False Positive Rate (FPR) and False Negative Rate (FNR)
    between male and female applicants.
    """

    print("\n" + "=" * 60)
    print("Fairness Analysis - Separation Criterion")
    print("=" * 60)

    fairness_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "sex": sex_test,
    })

    results = {}

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

        results[group] = {
            "samples": len(group_df),
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "true_positive_rate": tpr,
        }

        print(f"\nGroup: {group}")
        print(f"Samples: {len(group_df)}")
        print(f"False Positive Rate: {fpr:.4f}")
        print(f"False Negative Rate: {fnr:.4f}")
        print(f"True Positive Rate : {tpr:.4f}")

    if "male" in results and "female" in results:
        fpr_difference = abs(
            results["male"]["false_positive_rate"]
            - results["female"]["false_positive_rate"]
        )

        fnr_difference = abs(
            results["male"]["false_negative_rate"]
            - results["female"]["false_negative_rate"]
        )

        separation_difference = (fpr_difference + fnr_difference) / 2

        results["separation"] = {
            "fpr_difference": fpr_difference,
            "fnr_difference": fnr_difference,
            "separation_difference": separation_difference,
        }

        print("\n" + "-" * 60)
        print("Separation Differences")
        print("-" * 60)
        print(f"FPR Difference       : {fpr_difference:.4f}")
        print(f"FNR Difference       : {fnr_difference:.4f}")
        print(f"Separation Difference: {separation_difference:.4f}")

        if separation_difference > 0.05:
            raise ValueError(
                f"Fairness Gate failed: Separation difference ({separation_difference:.2%}) "
                f"exceeds the allowed threshold of 5.00%. "
                f"FPR difference: {fpr_difference:.2%}, "
                f"FNR difference: {fnr_difference:.2%}."
            )

    else:
        raise ValueError(
            "Fairness Analysis failed: both groups 'male' and 'female' must be present."
        )

    print("\nFairness Analysis completed.")
    print("=" * 60)

    return results