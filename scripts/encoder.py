from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

def create_random_forest_preprocessor(X):
    """
    Creates the preprocessing pipeline for the Random Forest model.

    The same preprocessor should be used in:
    - hyperparameter tuning
    - final model training
    - model validation
    - test evaluation
    """

    ordinal_features = [
        "employment",
        "job",
    ]

    binary_features = [
        "has_additional_security",
        "high_risk_financial_status",
    ]

    nominal_features = [
        "purpose",
        "credit_history",
        "other_parties",
        "property_magnitude",
        "other_payment_plans",
        "housing",
        "age_group",
        "checking_status",
        "savings_status",
    ]

    numeric_features = X.select_dtypes(
        include=["int64", "int32", "float64", "float32"]
    ).columns.tolist()
    print(numeric_features)

    numeric_features = [
        col for col in numeric_features
        if col not in binary_features
    ]

    employment_categories = [
        ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"]
    ]

    job_categories = [
        [
            "unemp/unskilled non res",
            "unskilled resident",
            "skilled",
            "high qualif/self emp/mgmt",
        ]
    ]

    ordinal_categories = employment_categories + job_categories

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_features,
            ),
            (
                "nominal",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                nominal_features,
            ),
            (
                "binary",
                "passthrough",
                binary_features,
            ),
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor