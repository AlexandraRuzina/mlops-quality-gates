import pandas as pd
import great_expectations as gx
from zenml import step
from zenml.logger import get_logger
from typing import Annotated


logger = get_logger(__name__)


@step
def data_validation_gate(df: pd.DataFrame) -> Annotated[pd.DataFrame, "validated_data"]:
    """
    Data Validation Gate:
    Prüft, ob der Rohdatensatz grundsätzlich für die weitere Verarbeitung geeignet ist.
    """

    failed_expectations = []

    # 1. Datensatz vorhanden und geladen
    if df is None or df.empty:
        error_message = (
            "Data Validation Gate failed:\n"
            "- Dataset is empty or not loaded."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # Great Expectations-Kontext erstellen
    context = gx.get_context(mode="ephemeral")

    # Die Datenquelle ist ein Pandas-DataFrame
    data_source = context.data_sources.add_pandas("pandas")

    # Registrierung Data Asset
    data_asset = data_source.add_dataframe_asset(name="dataco_raw_data")

    # Gesamten DataFrame als eine Einheit prüfen
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "dataco_raw_batch"
    )

    # DataFrame df an Great Expectations übergeben
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    # Expectation Suite ist eine Sammlung von Prüfregeln
    suite = gx.ExpectationSuite(name="data_preparation_suite")

    # Datensatz wird mit Prüfregelsammlung verbunden
    validator = context.get_validator(
        batch=batch,
        expectation_suite=suite,
    )

    # 2. Ausreichende Datenmenge
    validator.expect_table_row_count_to_be_between(
        min_value=1000,
        max_value=None,
    )

    # 3. Zielvariable vorhanden
    validator.expect_column_to_exist("class")

    for column in df.columns:
        validator.expect_column_values_to_not_be_null(
            column=column,
            mostly=0.95,
        )

    # Validation ausführen
    result = validator.validate()

    for expectation in result.results:
        if not expectation.success:
            config = expectation.expectation_config
            column = config.kwargs.get("column", "unknown_column")

            failed_expectations.append(
                f"{config.type} failed for column: {column}"
            )

    # 5. Klassenverteilung prüfen
    # Nur prüfen, wenn die Zielvariable tatsächlich vorhanden ist.
    if "class" in df.columns:
        class_distribution = df["class"].value_counts(normalize=True)

        if class_distribution.min() < 0.1:
            failed_expectations.append(
                "Class distribution is too imbalanced: "
                f"{class_distribution.to_dict()}"
            )

    if failed_expectations:
        error_message = (
            "Data Validation Gate failed. Failed expectations:\n"
            + "\n".join(f"- {failure}" for failure in failed_expectations)
        )

        logger.error(error_message)
        raise ValueError(error_message)

    logger.info("Data Validation Gate passed.")
    print("Data Validation Gate passed.")

    return df

