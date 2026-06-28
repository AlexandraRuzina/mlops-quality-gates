from zenml import step
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Annotated


@step
def train_test_split_step(
    df: pd.DataFrame,
) -> tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
    Annotated[pd.Series, "sex_test"],
]:
    """
    Teilt den Datensatz in Trainings- und Testdaten (80:20) auf.
    Die Klassenverteilung der Zielvariable bleibt erhalten.
    Das sensitive Attribut 'sex' wird separat mitgesplittet,
    aber nicht als Trainingsfeature verwendet.
    """

    target_column = "class"
    sensitive_column = "sex"

    y = df[target_column]
    sex = df[sensitive_column]

    X = df.drop(columns=[target_column, sensitive_column])

    X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
        X,
        y,
        sex,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Trainingsdaten: {X_train.shape[0]} Zeilen")
    print(f"Testdaten: {X_test.shape[0]} Zeilen")
    print(f"Sensitive Attribut Trainingsdaten: {sex_train.shape[0]} Zeilen")
    print(f"Sensitive Attribut Testdaten: {sex_test.shape[0]} Zeilen")

    return X_train, X_test, y_train, y_test, sex_test