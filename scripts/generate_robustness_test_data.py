import numpy as np
import pandas as pd


def generate_perturbation(
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Creates a perturbed version of the test set for robustness evaluation.

    Numerical features are randomly perturbed by ±5% while preserving
    domain constraints.
    """

    # Reproducible perturbations
    #Zufallsgenerator
    #gleichen Zufallswerte bei jedem Lauf --> Reproduzierbarkeit
    rng = np.random.default_rng(seed=42)

    X_test_perturbed = X_test.copy()

    numeric_features = [
        "duration",
        "credit_amount",
    ]

    for feature in numeric_features:
        if feature in X_test_perturbed.columns:

            # Random factor between 0.95 and 1.05
            # Für jede Zeile wird ein zufälliger Faktor zwischen 0.95 und 1.05 erzeugt.
            #0.95 = Wert wird um 5 % kleiner
            #1.00 = Wert bleibt gleich
            #1.05 = Wert wird um 5 % größer
            noise = rng.uniform(
                low=0.95,
                high=1.05,
                size=len(X_test_perturbed),
            )

            #Hier wird das jeweilige Feature tatsächlich verändert. Jeder Wert wird mit seinem zufälligen Faktor multipliziert.
            X_test_perturbed[feature] = (
                X_test_perturbed[feature] * noise
            )

    # ------------------------------------------------------------------
    # Enforce domain constraints
    # ------------------------------------------------------------------

    X_test_perturbed["duration"] = (
        X_test_perturbed["duration"]
        .round()
        .clip(lower=0)
    )

    X_test_perturbed["credit_amount"] = (
        X_test_perturbed["credit_amount"].clip(lower=0)
    )

    X_test_perturbed["monthly_payment"] = (
            X_test_perturbed["credit_amount"]
            / X_test_perturbed["duration"]
    ).clip(lower=0)

    return X_test_perturbed

def generate_inflation_test_data(
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Creates an inflation stress test dataset.

    Simulates a 3% inflation scenario by increasing the requested
    credit amount by 3% and recalculating the monthly payment.
    """

    X_test_inflation = X_test.copy()

    # Increase credit amount by 3%
    X_test_inflation["credit_amount"] = (
        X_test_inflation["credit_amount"] * 1.03
    )

    # Recalculate monthly payment
    X_test_inflation["monthly_payment"] = (
        X_test_inflation["credit_amount"]
        / X_test_inflation["duration"]
    )

    return X_test_inflation