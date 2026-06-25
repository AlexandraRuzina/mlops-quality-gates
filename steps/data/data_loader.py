from pathlib import Path
import pandas as pd
from zenml import step


@step
def load_data() -> pd.DataFrame:
    """Lädt den DataCo-Datensatz aus einer Parquet-Datei."""

    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / "data" / "raw" / "german_credit.parquet"

    df = pd.read_parquet(data_path)

    print(f"Datensatz geladen: {df.shape}")

    return df