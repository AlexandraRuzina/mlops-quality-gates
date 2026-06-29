from typing import Any, Annotated

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
) -> tuple[
    Annotated[dict, "metamorphic_report"],
    Annotated[bool, "metamorphic_gate_passed"],
]:
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

        passed = violation_rate <= MAX_VIOLATION_RATE

        return {
            "number_of_cases": number_of_cases,
            "number_of_violations": number_of_violations,
            "violation_rate": violation_rate,
            "threshold": MAX_VIOLATION_RATE,
            "comparison": "<=",
            "passed": passed,
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

    gate_passed = all(
        result["passed"] for result in relation_results.values()
    )

    metamorphic_report = {
        "gate_name": "Metamorphic Verification Gate",
        "gate_passed": gate_passed,
        "max_violation_rate": MAX_VIOLATION_RATE,
        "metrics": relation_results,
    }

    print("\n" + "=" * 60)
    print("Metamorphic Verification Gate")
    print("=" * 60)

    for relation_name, result in relation_results.items():
        status = "PASSED" if result["passed"] else "FAILED"

        print(
            f"{relation_name:<32} "
            f"violations={result['number_of_violations']}/"
            f"{result['number_of_cases']} "
            f"violation_rate={result['violation_rate']:.4f} "
            f"threshold <= {result['threshold']:.4f} "
            f"{status}"
        )

    print("-" * 60)
    print(f"Overall Gate Status: {'PASSED' if gate_passed else 'FAILED'}")
    print("=" * 60)

    return metamorphic_report, gate_passed