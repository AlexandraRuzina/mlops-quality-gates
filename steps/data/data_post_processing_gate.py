from zenml import step
from zenml.logger import get_logger
import pandas as pd
import great_expectations as gx
from typing import Annotated


logger = get_logger(__name__)


@step
def data_post_processing_gate(df: pd.DataFrame) -> Annotated[pd.DataFrame, "post_validated_data"]:
    """
    Prüft den finalen Datensatz nach Datenvorverarbeitung und Feature Engineering.
    Wenn eine Prüfung fehlschlägt, wird die Pipeline abgebrochen.
    """

    target_column = "class"
    failed_expectations = []

    expected_features = [
        "monthly_payment",
        "age_group",
        "has_additional_security",
        "high_risk_financial_status",
    ]

    deleted_attributes = [
        "personal_status",
        "foreign_worker",
        "own_telephone",
        "age",
    ]

    min_rows = 800

    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas("post_processed_data_source")
    data_asset = data_source.add_dataframe_asset(name="post_processed_data_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "post_processed_batch"
    )

    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = context.suites.add(
        gx.ExpectationSuite(name="data_post_processing_suite")
    )

    validator = context.get_validator(
        batch=batch,
        expectation_suite=suite,
    )

    # 1. Ausreichende Datenmenge vorhanden
    validator.expect_table_row_count_to_be_between(
        min_value=min_rows
    )

    # 3. Zielvariable darf nur 0 und 1 enthalten
    if target_column in df.columns:
        validator.expect_column_values_to_be_in_set(
            column=target_column,
            value_set=[0, 1],
        )

        # 4. Klassenverteilung weiterhin brauchbar
        class_distribution = df[target_column].value_counts(normalize=True)

        if class_distribution.min() < 0.1:
            failed_expectations.append(
                "Class distribution is too imbalanced: "
                f"{class_distribution.to_dict()}"
            )
    else:
        failed_expectations.append(
            f"Target column '{target_column}' is missing."
        )

    # 5. Erwartete Features vorhanden
    for feature in expected_features:
        validator.expect_column_to_exist(feature)

    # 6. Keine Nullwerte in erzeugten Features
    for feature in expected_features:
        if feature in df.columns:
            validator.expect_column_values_to_not_be_null(feature)
        else:
            failed_expectations.append(
                f"Expected feature '{feature}' is missing."
            )

    # 7. Datentypen und Wertebereiche prüfen
    if "monthly_payment" in df.columns:
        validator.expect_column_values_to_be_between(
            column="monthly_payment",
            min_value=0,
            strict_min=True,
        )

    if "has_additional_security" in df.columns:
        validator.expect_column_values_to_be_in_set(
            column="has_additional_security",
            value_set=[0, 1],
        )

    if "high_risk_financial_status" in df.columns:
        validator.expect_column_values_to_be_in_set(
            column="high_risk_financial_status",
            value_set=[0, 1],
        )

    if "age_group" in df.columns:
        validator.expect_column_values_to_be_in_set(
            column="age_group",
            value_set=["<25", "25-40", "40-60", "60+"],
        )

    # 8. Sensible Attribute dürfen nicht mehr enthalten sein
    remaining_deleted_attributes = [
        col for col in deleted_attributes if col in df.columns
    ]

    if remaining_deleted_attributes:
        failed_expectations.append(
            "Sensitive/deleted attributes are still present: "
            f"{remaining_deleted_attributes}"
        )

    # Great Expectations Validation ausführen
    result = validator.validate()

    for validation_result in result.results:
        if not validation_result.success:
            expectation_type = validation_result.expectation_config.type
            column = validation_result.expectation_config.kwargs.get(
                "column",
                "table",
            )

            failed_expectations.append(
                f"{expectation_type} failed for column: {column}"
            )

    if failed_expectations:
        error_message = (
            "Data Post-Processing Gate failed. Failed expectations:\n"
            + "\n".join(f"- {failure}" for failure in failed_expectations)
        )

        logger.error(error_message)
        raise ValueError(error_message)

    logger.info("Data Post-Processing Gate passed.")
    print("Data Post-Processing Gate passed.")

    return df