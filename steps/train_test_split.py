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
]:
    """
    Teilt den Datensatz in Trainings- und Testdaten (80:20) auf.
    Die Klassenverteilung der Zielvariable bleibt erhalten.
    """

    target_column = "class"

    # Features und Zielvariable trennen
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 80:20 Split
    #random_state=42 Split ist reproduzierbar, bei jedem Pipeline -Lauf selbe Aufteilung
    #Klassenverteilung bleibt erhalten --> stratify=y
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Trainingsdaten: {X_train.shape[0]} Zeilen")
    print(f"Testdaten: {X_test.shape[0]} Zeilen")

    return X_train, X_test, y_train, y_test