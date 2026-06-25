import pandas as pd
import great_expectations as gx
from zenml import step


@step
def data_validation_gate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data Validation Gate:
    Prüft, ob der Rohdatensatz grundsätzlich für die weitere Verarbeitung geeignet ist.
    """

    # 1. Datensatz vorhanden und geladen
    if df is None or df.empty:
        raise ValueError("Data Validation Gate failed: Dataset is empty or not loaded.")

    # Great Expectations DataFrame-Validator
    # Great Expectations-Kontext erstellen
    # ephemeral = temporär, nur für den Pipeline-Lauf
    context = gx.get_context(mode="ephemeral")

    #Die Datenquelle ist ein Pandas-DataFrame
    data_source = context.data_sources.add_pandas("pandas")
    #Registrierung Data Asset --> Datensatz den man prüfen möchte
    data_asset = data_source.add_dataframe_asset(name="dataco_raw_data")
    #gesamten Dataframe als eine Einheit prüfen
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "dataco_raw_batch"
    )

    #DataFrame df an Great Expectations übergeben
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    # Expectation Suite ist eine Sammlung von Prüfregeln
    # Sammlung der Regeln für das Data Validation Gate
    suite = gx.ExpectationSuite(name="data_preparation_suite")
    #Datensatz wird mit Prüfregelsammlung verbunden
    # Mit Validator formuliert man Prüfungen
    validator = context.get_validator(
        batch=batch,
        expectation_suite=suite,
    )

    # 2. Ausreichende Datenmenge
    # Datensatz muss mindestns 1000 Zeilen haben, es gibt keine Obergrenze
    validator.expect_table_row_count_to_be_between(
        min_value=1000,
        max_value=None
    )

    # 3. Zielvariable vorhanden
    validator.expect_column_to_exist("class")

    # 4. Nullwertanteil nicht zu hoch in der Zielvariablen Spalte
    # Mindestens 80% der Werte dürfen nicht null sein --> Nullwertanteil nicht zu hoch --> Nicht Null sondern NaN
    validator.expect_column_values_to_not_be_null(
        column="class",
        mostly=0.8
    )

    # Berechnung der Klassenverteilung der Zielvariablen
    class_distribution = df["class"].value_counts(normalize=True)

    #Macht die kleinere Klasse weniger als 10% aus
    if class_distribution.min() < 0.1:
        raise ValueError(
            f"Data Validation Gate failed: Class distribution is too imbalanced: "
            f"{class_distribution.to_dict()}"
        )

    # Validation ausführen
    # Ausführung aller Great-Expectations-Regeln
    result = validator.validate()

    failed_expectations = []

    for expectation in result.results:
        if not expectation.success:
            config = expectation.expectation_config
            column = config.kwargs.get("column", "unknown_column")
            failed_expectations.append(
                f"{config.type} failed for column: {column}"
            )

    if failed_expectations:
        raise ValueError(
            "Failed expectations:\n"
            + "\n".join(failed_expectations)
        )

    print("Data Validation Gate passed.")

    return df
