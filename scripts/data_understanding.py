from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================
DATA_PATH = Path("../data/raw/german_credit.parquet")
IMAGE_DIR = Path("../images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Load data
# =========================
df = pd.read_parquet(DATA_PATH)

print("Columns:")
for col in df.columns:
    print(col)

print("\nTarget distribution:")
print(df["class"].value_counts())


# =========================
# Helper: gender extraction
# =========================
def extract_gender(value: str) -> str:
    """
    Extracts gender from the German Credit variable 'personal_status'.

    In the original German Credit dataset, personal_status combines
    marital status and sex. Depending on the dataset version, the values
    can either be original codes such as A91-A95 or readable strings.
    """
    value = str(value).strip().lower()

    # Original German Credit codes
    if value in ["a91", "a93", "a94"]:
        return "male"
    if value in ["a92", "a95"]:
        return "female"

    # Readable OpenML values
    # Important: check "female" before "male", because "female" contains "male".
    if "female" in value:
        return "female"
    if "male" in value:
        return "male"

    return "unknown"


# personal_status = personal status and sex
df["gender"] = df["personal_status"].apply(extract_gender)

print("\nGender distribution:")
print(df["gender"].value_counts())


# =========================
# 1. Target distribution
# =========================
plt.figure(figsize=(7, 5))
df["class"].value_counts().plot(kind="bar")
plt.title("Distribution of Credit Risk Classes")
plt.xlabel("Credit class")
plt.ylabel("Number of applicants")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "01_target_distribution.png", dpi=300)
plt.close()


# =========================
# 2. Gender distribution
# =========================
plt.figure(figsize=(7, 5))
df["gender"].value_counts().plot(kind="bar")
plt.title("Gender Distribution in the Dataset")
plt.xlabel("Gender")
plt.ylabel("Number of applicants")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "02_gender_distribution.png", dpi=300)
plt.close()


# =========================
# 3. Credit class by gender (normalized)
# =========================
gender_credit_distribution = pd.crosstab(
    df["gender"],
    df["class"],
    normalize="index"
) * 100

gender_credit_distribution.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Credit Class Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Share within gender group (%)")
plt.legend(title="Credit class")
plt.tight_layout()
plt.savefig(
    IMAGE_DIR / "03_credit_class_by_gender_normalized.png",
    dpi=300
)
plt.close()


# =========================
# 4. Age distribution by credit class
# =========================
plt.figure(figsize=(8, 5))

for credit_class in sorted(df["class"].unique()):
    subset = df[df["class"] == credit_class]
    plt.hist(
        subset["age"],
        bins=20,
        alpha=0.6,
        label=str(credit_class)
    )

plt.title("Age Distribution by Credit Class")
plt.xlabel("Age")
plt.ylabel("Number of applicants")
plt.legend(title="Credit class")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "04_age_distribution_by_class.png", dpi=300)
plt.close()


# =========================
# 5. Credit amount distribution by credit class
# =========================
plt.figure(figsize=(8, 5))
df.boxplot(column="credit_amount", by="class")
plt.title("Credit Amount by Credit Class")
plt.suptitle("")
plt.xlabel("Credit class")
plt.ylabel("Credit amount")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "05_credit_amount_by_class.png", dpi=300)
plt.close()

numeric_columns = [
    "duration",
    "residence_since",
    "age"
]

df[numeric_columns].boxplot(figsize=(12, 6))
plt.title("Analysis of Outliers")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "06_outliers", dpi=300)
plt.close()

df.boxplot(column="credit_amount")
plt.title("Credit Amount")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "07_credit_amount", dpi=300)
plt.close()

print(f"\nPlots saved in: {IMAGE_DIR.resolve()}")