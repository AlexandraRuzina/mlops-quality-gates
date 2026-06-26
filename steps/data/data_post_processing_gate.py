from zenml import step
import pandas as pd
import great_expectations as gx


@step
def data_post_processing_gate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prüft den finalen Datensatz nach Datenvorverarbeitung und Feature Engineering.
    Wenn eine Prüfung fehlschlägt, wird die Pipeline abgebrochen.
    """

    target_column = "class"

    expected_features = [
        "monthly_payment",
        "age_group",
        "has_additional_security",
        "high_risk_financial_status"
    ]

    deleted_attributes = [
        "personal_status",
        "foreign_worker",
        "own_telephone",
        "age"
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

    # 5. Klassenverteilung weiterhin brauchbar
    class_distribution = df[target_column].value_counts(normalize=True)

    if class_distribution.min() < 0.1:
        raise ValueError(
            f"Data Post-Processing Gate  failed: Class distribution is too imbalanced: "
            f"{class_distribution.to_dict()}"
        )

    # 6. Erwartete Features vorhanden
    for feature in expected_features:
        validator.expect_column_to_exist(feature)

    # 7. Keine Nullwerte in erzeugten Features
    for feature in expected_features:
        validator.expect_column_values_to_not_be_null(feature)

    # 8. Datentypen und Wertebereiche prüfen

    validator.expect_column_values_to_be_between(
        column="monthly_payment",
        min_value=0,
        strict_min=True
    )

    validator.expect_column_values_to_be_in_set(
        column="has_additional_security",
        value_set=[0, 1]
    )

    validator.expect_column_values_to_be_in_set(
        column="high_risk_financial_status",
        value_set=[0, 1],
    )

    validator.expect_column_values_to_be_in_set(
        column="age_group",
        value_set=["<25", "25-40", "40-60", "60+"]
    )

    # 9.Attribute dürfen nicht mehr enthalten sein
    remaining_deleted_attributes = [
        col for col in deleted_attributes if col in df.columns
    ]

    if remaining_deleted_attributes:
        raise ValueError(
            f"Data Post-Processing Gate failed: "
            f"Sensible Attribute sind noch vorhanden: {remaining_deleted_attributes}"
        )

    result = validator.validate()

    if not result.success:
        failed_expectations = []

        for validation_result in result.results:
            if not validation_result.success:
                expectation_type = validation_result.expectation_config.type
                column = validation_result.expectation_config.kwargs.get("column", "table")

                failed_expectations.append(
                    f"{expectation_type} failed for column: {column}"
                )

        raise ValueError(
            "Data Post-Processing Gate failed:\n"
            + "\n".join(failed_expectations)
        )

    print("Data Post-Processing Gate passed.")

    return df