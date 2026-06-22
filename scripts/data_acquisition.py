from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


# =========================
# Dataset configuration
# =========================
DATASET_ID = 31  # OpenML credit-g dataset

OUTPUT_DIR = Path("../data/raw")

PARQUET_FILE = OUTPUT_DIR / "german_credit.parquet"
CSV_FILE = OUTPUT_DIR / "german_credit.csv"


# =========================
# Load dataset from OpenML
# =========================
credit_data = fetch_openml(
    data_id=DATASET_ID,
    as_frame=True,
    parser="auto"
)

features = credit_data.data
target = credit_data.target.rename("class")

df = pd.concat([features, target], axis=1)


# =========================
# Save raw dataset
# =========================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_parquet(PARQUET_FILE, index=False)
df.to_csv(CSV_FILE, index=False)


# =========================
# Basic information
# =========================
print(f"Parquet saved to: {PARQUET_FILE}")
print(f"CSV saved to: {CSV_FILE}")
print(f"Shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

print("\nTarget distribution:")
print(df["class"].value_counts())