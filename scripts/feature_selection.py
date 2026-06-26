from zenml.client import Client
import pandas as pd
import matplotlib.pyplot as plt

client = Client()

artifact_x = client.get_artifact_version(name_id_or_prefix="X_train")
X_train = artifact_x.load()

print("X_train Shape:", X_train.shape)

# Nur numerische Features betrachten
numeric_features = X_train.select_dtypes(include=["int64", "int32", "float64", "float32"])

# Korrelationsmatrix berechnen
correlation_matrix = numeric_features.corr()

print("\nKorrelationsmatrix:")
print(correlation_matrix)

# Starke Korrelationen finden
threshold = 0.8

high_correlations = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        feature_1 = correlation_matrix.columns[i]
        feature_2 = correlation_matrix.columns[j]
        correlation_value = correlation_matrix.iloc[i, j]

        if abs(correlation_value) >= threshold:
            high_correlations.append(
                {
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": correlation_value,
                }
            )

high_correlations_df = pd.DataFrame(high_correlations)

print(f"\nStark korrelierte Feature-Paare |r| >= {threshold}:")
if high_correlations_df.empty:
    print("Keine stark korrelierten Feature-Paare gefunden.")
else:
    print(high_correlations_df.sort_values(by="correlation", ascending=False))


plt.figure(figsize=(12, 10))

plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="Korrelationskoeffizient")

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=90,
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
)

# Korrelationswerte einzeichnen
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
        )

plt.title("Korrelationsmatrix der numerischen Features")
plt.tight_layout()
plt.show()

