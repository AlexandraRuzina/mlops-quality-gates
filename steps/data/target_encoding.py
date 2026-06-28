from zenml import step
import pandas as pd
from typing import Annotated


@step
def encode_target(df: pd.DataFrame) -> Annotated[pd.DataFrame, "encoded_target_data"]:
    """
    Kodiert die Zielvariable class binär.
    good = 0
    bad = 1
    """

    df = df.copy()

    target_column = "class"

    target_mapping = {
        "good": 0,
        "bad": 1,
    }

    df[target_column] = df[target_column].map(target_mapping)

    if df[target_column].isnull().any():
        invalid_values = df.loc[df[target_column].isnull(), target_column].unique()
        raise ValueError(
            f"Target Encoding fehlgeschlagen. Ungültige Werte: {invalid_values}"
        )

    df[target_column] = df[target_column].astype(int)

    print("Target Encoding abgeschlossen.")
    print(df[target_column].value_counts())

    return df