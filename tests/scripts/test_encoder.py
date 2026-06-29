import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from scripts.encoder import create_random_forest_preprocessor


@pytest.fixture
def sample_X():
    return pd.DataFrame(
        [
            {
                "employment": "1<=X<4",
                "job": "skilled",
                "has_additional_security": 0,
                "high_risk_financial_status": 1,
                "purpose": "new car",
                "credit_history": "existing paid",
                "other_parties": "none",
                "property_magnitude": "car",
                "other_payment_plans": "none",
                "housing": "own",
                "age_group": "25-40",
                "checking_status": "<0",
                "savings_status": "<100",
                "duration": 24,
                "credit_amount": 3000,
                "installment_commitment": 2,
                "residence_since": 2,
                "existing_credits": 1,
                "num_dependents": 1,
                "monthly_payment": 125.0,
            },
            {
                "employment": ">=7",
                "job": "high qualif/self emp/mgmt",
                "has_additional_security": 1,
                "high_risk_financial_status": 0,
                "purpose": "business",
                "credit_history": "critical/other existing credit",
                "other_parties": "guarantor",
                "property_magnitude": "real estate",
                "other_payment_plans": "bank",
                "housing": "own",
                "age_group": "40-60",
                "checking_status": ">=200",
                "savings_status": ">=1000",
                "duration": 36,
                "credit_amount": 6000,
                "installment_commitment": 3,
                "residence_since": 4,
                "existing_credits": 2,
                "num_dependents": 2,
                "monthly_payment": 166.67,
            },
        ]
    )


def test_create_random_forest_preprocessor_returns_column_transformer(sample_X):
    preprocessor = create_random_forest_preprocessor(sample_X)

    assert isinstance(preprocessor, ColumnTransformer)


def test_preprocessor_contains_expected_transformers(sample_X):
    preprocessor = create_random_forest_preprocessor(sample_X)

    transformer_names = [
        name for name, _, _ in preprocessor.transformers
    ]

    assert transformer_names == [
        "ordinal",
        "nominal",
        "binary",
        "numeric",
    ]


def test_preprocessor_uses_expected_ordinal_features(sample_X):
    preprocessor = create_random_forest_preprocessor(sample_X)

    ordinal_features = preprocessor.transformers[0][2]

    assert ordinal_features == [
        "employment",
        "job",
    ]


def test_preprocessor_uses_expected_binary_features(sample_X):
    preprocessor = create_random_forest_preprocessor(sample_X)

    binary_features = preprocessor.transformers[2][2]

    assert binary_features == [
        "has_additional_security",
        "high_risk_financial_status",
    ]


def test_preprocessor_uses_expected_nominal_features(sample_X):
    preprocessor = create_random_forest_preprocessor(sample_X)

    nominal_features = preprocessor.transformers[1][2]

    expected_nominal_features = [
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

    assert nominal_features == expected_nominal_features


def test_numeric_features_exclude_binary_features(sample_X):
    preprocessor = create_random_forest_preprocessor(sample_X)

    numeric_features = preprocessor.transformers[3][2]

    assert "duration" in numeric_features
    assert "credit_amount" in numeric_features
    assert "monthly_payment" in numeric_features

    assert "has_additional_security" not in numeric_features
    assert "high_risk_financial_status" not in numeric_features


def test_preprocessor_can_fit_transform_valid_data(sample_X):
    preprocessor = create_random_forest_preprocessor(sample_X)

    transformed = preprocessor.fit_transform(sample_X)

    assert transformed.shape[0] == len(sample_X)
    assert transformed.shape[1] > 0


def test_preprocessor_handles_unknown_nominal_category(sample_X):
    preprocessor = create_random_forest_preprocessor(sample_X)
    preprocessor.fit(sample_X)

    new_data = sample_X.copy()
    new_data.loc[0, "purpose"] = "unknown purpose"

    transformed = preprocessor.transform(new_data)

    assert transformed.shape[0] == len(new_data)


def test_preprocessor_handles_unknown_ordinal_category(sample_X):
    preprocessor = create_random_forest_preprocessor(sample_X)
    preprocessor.fit(sample_X)

    new_data = sample_X.copy()
    new_data.loc[0, "employment"] = "unknown employment"

    transformed = preprocessor.transform(new_data)

    assert transformed.shape[0] == len(new_data)


def test_unknown_ordinal_category_is_encoded_as_minus_one(sample_X):
    preprocessor = create_random_forest_preprocessor(sample_X)
    preprocessor.fit(sample_X)

    new_data = sample_X.copy()
    new_data.loc[0, "employment"] = "unknown employment"

    transformed = preprocessor.transform(new_data)

    assert transformed[0, 0] == -1