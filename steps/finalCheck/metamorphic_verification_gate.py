from typing import Any

import numpy as np
import pandas as pd
from zenml import step

from scripts.generate_metamorphic_test_data import (
    generate_duration_increase_data,
    generate_checking_status_improvement_data,
    generate_savings_status_deterioration_data,
)


@step
def metamorphic_verification_gate(
    model: Any,
    X_test: pd.DataFrame,
    y_proba,
) -> dict:
    """
    Verifies model behavior using metamorphic relations.

    Assumption:
    - good = 0
    - bad = 1

    Therefore, P(bad) is located at index 1 of predict_proba output.
    """

    MAX_VIOLATION_RATE = 0.1
    BAD_CLASS_INDEX = 1

    p_bad_original = y_proba

    relation_results = {}

    def evaluate_relation(
        relation_name: str,
        X_transformed: pd.DataFrame,
        mask: pd.Series,
        expected_direction: str,
    ) -> dict:
        transformed_proba = model.predict_proba(X_transformed)
        p_bad_transformed = transformed_proba[:, BAD_CLASS_INDEX]

        original_values = p_bad_original[mask]
        transformed_values = p_bad_transformed[mask]

        if expected_direction == "increase":
            violations = transformed_values < original_values

        elif expected_direction == "decrease":
            violations = transformed_values > original_values

        else:
            raise ValueError(
                f"Unknown expected direction: {expected_direction}"
            )

        number_of_cases = len(original_values)
        number_of_violations = int(np.sum(violations))

        violation_rate = (
            number_of_violations / number_of_cases
            if number_of_cases > 0
            else 0.0
        )

        return {
            "number_of_cases": number_of_cases,
            "number_of_violations": number_of_violations,
            "violation_rate": violation_rate,
        }

    all_rows_mask = pd.Series(True, index=X_test.index)

    # MR1: Higher duration -> P(bad) should not decrease
    X_duration = generate_duration_increase_data(X_test)

    relation_results["duration_increase"] = evaluate_relation(
        relation_name="duration_increase",
        X_transformed=X_duration,
        mask=all_rows_mask,
        expected_direction="increase",
    )

    # MR2: Better checking status -> P(bad) should not increase
    X_checking, checking_mask = generate_checking_status_improvement_data(
        X_test
    )

    relation_results["checking_status_improvement"] = evaluate_relation(
        relation_name="checking_status_improvement",
        X_transformed=X_checking,
        mask=checking_mask,
        expected_direction="decrease",
    )

    # MR3: Lower savings status -> P(bad) should not decrease
    X_savings, savings_mask = generate_savings_status_deterioration_data(
        X_test
    )

    relation_results["savings_status_deterioration"] = evaluate_relation(
        relation_name="savings_status_deterioration",
        X_transformed=X_savings,
        mask=savings_mask,
        expected_direction="increase",
    )

    failed_relations = {
        name: result
        for name, result in relation_results.items()
        if result["violation_rate"] > MAX_VIOLATION_RATE
    }

    if failed_relations:
        raise ValueError(
            f"Metamorphic Verification Gate failed: "
            f"{len(failed_relations)} relation(s) exceeded the maximum "
            f"allowed violation rate of {MAX_VIOLATION_RATE:.2f}. "
            f"Failed relations: {failed_relations}"
        )

    return {
        "max_violation_rate": MAX_VIOLATION_RATE,
        "relation_results": relation_results,
        "gate_passed": True,
    }