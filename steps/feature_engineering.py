from zenml import step
import pandas as pd
from pathlib import Path


@step
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    return df