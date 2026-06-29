from zenml import step
from typing import Annotated

@step
def production_readiness_gate(
    production_readiness_passed: bool,
) -> Annotated[bool, "production_readiness_checked"]:
    """
    Final production readiness decision.

    Stops the pipeline if one or more quality gates failed.
    """

    if not production_readiness_passed:
        raise ValueError(
            "Production Readiness Gate failed. "
            "At least one quality gate did not pass. "
            "The model will not be registered."
        )

    print("Production Readiness Gate passed.")
    return True