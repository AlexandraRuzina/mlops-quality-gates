from pathlib import Path
import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("shashwatwork/dataco-smart-supply-chain-for-big-data-analysis")

# Hauptdatei
csv_file = Path(path) / "DataCoSupplyChainDataset.csv"

# Data-Verzeichnis anlegen
Path("../data/raw").mkdir(parents=True, exist_ok=True)

# CSV laden
df = pd.read_csv(csv_file, encoding="latin1")

# Als Parquet speichern
parquet_file = Path("../data/raw/dataco.parquet")
df.to_parquet(parquet_file, index=False)