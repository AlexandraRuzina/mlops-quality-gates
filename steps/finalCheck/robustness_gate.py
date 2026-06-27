from scripts.generate_robustness_test_data import generate_perturbation, generate_inflation_test_data
from typing import Any

import pandas as pd
from sklearn.metrics import f1_score
from zenml import step


@step
def robustness_gate(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_metrics: dict,
) -> dict:
    """
    Evaluates robustness using perturbation and inflation test data.
    """

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

    if mean_robustness_score < 0.95:
        raise ValueError(
            f"Robustness Gate failed: Mean robustness score "
            f"({mean_robustness_score:.3f}) is below the required threshold "
            f"of 0.95. The model is not sufficiently robust against the "
            f"evaluated perturbation scenarios."
        )

    return {
        "f1_clean": f1_clean,
        "f1_perturbation": f1_perturbation,
        "f1_inflation": f1_inflation,
        "robustness_perturbation": robustness_perturbation,
        "robustness_inflation": robustness_inflation,
        "mean_robustness_score": mean_robustness_score,
    }