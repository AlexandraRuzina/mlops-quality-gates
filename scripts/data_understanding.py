from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

##Credit Amount im Verhältnis zu target

plt.figure(figsize=(8,6))

plt.hist(
    df[df["class"]=="good"]["credit_amount"],
    bins=30,
    alpha=0.6,
    label="good"
)

plt.hist(
    df[df["class"]=="bad"]["credit_amount"],
    bins=30,
    alpha=0.6,
    label="bad"
)

plt.xlabel("Credit Amount")
plt.ylabel("Anzahl")
plt.title("Verteilung des Kreditbetrags")
plt.legend()

plt.savefig(IMAGE_DIR / "08_credit_amount_target", dpi=300)
plt.close()

stats = df.groupby("class")["credit_amount"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

print(stats)

quartiles = (
    df.groupby("class")["credit_amount"]
      .quantile([0.25, 0.5, 0.75])
      .unstack()
)

quartiles.columns = ["Q1", "Median", "Q3"]

print(quartiles)

##Duration  im Verhältnis zu target

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="class",
    y="duration"
)

plt.xlabel("Klasse")
plt.ylabel("Duration (Monate)")
plt.title("Kreditlaufzeit nach Zielklasse")

plt.savefig(IMAGE_DIR / "09_duration_target", dpi=300)
plt.close()

stats = df.groupby("class")["duration"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

print(stats)

quartiles = (
    df.groupby("class")["duration"]
      .quantile([0.25, 0.5, 0.75])
      .unstack()
)

quartiles.columns = ["Q1", "Median", "Q3"]

print(quartiles)

#Employment und target

employment_order = [
    "unemployed",
    "<1",
    "1<=X<4",
    "4<=X<7",
    ">=7",
]

df_plot = df.copy()
df_plot["employment"] = pd.Categorical(
    df_plot["employment"],
    categories=employment_order,
    ordered=True,
)

employment_class = pd.crosstab(
    df_plot["employment"],
    df_plot["class"],
    normalize="index"
)

employment_class.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 6)
)

plt.xlabel("Beschäftigungsdauer")
plt.ylabel("Anteil")
plt.title("Relative Verteilung der Zielklassen nach Beschäftigungsdauer")
plt.xticks(rotation=0)
plt.legend(title="Klasse")

plt.savefig(IMAGE_DIR / "10_employment_target", dpi=300)
plt.close()

# Checking Status und target

checking_order = [
    "no checking",
    "<0",
    "0<=X<200",
    ">=200",
]

df_plot = df.copy()
df_plot["checking_status"] = pd.Categorical(
    df_plot["checking_status"],
    categories=checking_order,
    ordered=True,
)

checking_class = pd.crosstab(
    df_plot["checking_status"],
    df_plot["class"],
    normalize="index"
)

checking_class.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 6)
)

plt.xlabel("Kontostatus")
plt.ylabel("Anteil")
plt.title("Relative Verteilung der Zielklassen nach Kontostatus")
plt.xticks(
    ticks=range(len(checking_order)),
    labels=[
        "Kein Konto",
        "< 0 DM",
        "0–200 DM",
        "≥ 200 DM",
    ],
    rotation=0,
)
plt.legend(title="Klasse")

plt.tight_layout()
plt.savefig(IMAGE_DIR / "11_checking_status_target", dpi=300)
plt.close()

#Savings Status Target

import pandas as pd
import matplotlib.pyplot as plt

savings_order = [
    "no known savings",
    "<100",
    "100<=X<500",
    "500<=X<1000",
    ">=1000",
]

df_plot = df.copy()
df_plot["savings_status"] = pd.Categorical(
    df_plot["savings_status"],
    categories=savings_order,
    ordered=True,
)

savings_class = pd.crosstab(
    df_plot["savings_status"],
    df_plot["class"],
    normalize="index"
)

savings_class.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 6)
)

plt.xlabel("Sparguthaben")
plt.ylabel("Anteil")
plt.title("Relative Verteilung der Zielklassen nach Sparguthaben")

plt.xticks(
    ticks=range(len(savings_order)),
    labels=[
        "Kein Sparguthaben",
        "< 100 DM",
        "100–500 DM",
        "500–1000 DM",
        "≥ 1000 DM",
    ],
    rotation=0,
)

plt.legend(title="Klasse")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "12_savings_status_target", dpi=300)
plt.close()

#Job Target

import pandas as pd
import matplotlib.pyplot as plt

job_order = [
    "unemp/unskilled non res",
    "unskilled resident",
    "skilled",
    "high qualif/self emp/mgmt",
]

df_plot = df.copy()
df_plot["job"] = pd.Categorical(
    df_plot["job"],
    categories=job_order,
    ordered=True,
)

job_class = pd.crosstab(
    df_plot["job"],
    df_plot["class"],
    normalize="index"
)

job_class.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 6)
)

plt.xlabel("Beruf")
plt.ylabel("Anteil")
plt.title("Relative Verteilung der Zielklassen nach Berufsgruppe")

plt.xticks(
    ticks=range(len(job_order)),
    labels=[
        "Ungelernt\n(nicht ansässig)",
        "Ungelernt\n(ansässig)",
        "Facharbeiter",
        "Hochqualifiziert /\nSelbstständig /\nManagement",
    ],
    rotation=0,
)

plt.legend(title="Klasse")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "13_job_target", dpi=300)
plt.close()

installment_order = [1, 2, 3, 4]

df_plot = df.copy()
df_plot["installment_commitment"] = pd.Categorical(
    df_plot["installment_commitment"],
    categories=installment_order,
    ordered=True,
)

installment_class = pd.crosstab(
    df_plot["installment_commitment"],
    df_plot["class"],
    normalize="index"
)

installment_class.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 6)
)

plt.xlabel("Installment Commitment")
plt.ylabel("Anteil")
plt.title("Relative Verteilung der Zielklassen nach Installment Commitment")

plt.xticks(
    ticks=range(len(installment_order)),
    labels=[
        "1",
        "2",
        "3",
        "4",
    ],
    rotation=0,
)

plt.legend(title="Klasse")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "14_installment_rate_target", dpi=300)
plt.close()