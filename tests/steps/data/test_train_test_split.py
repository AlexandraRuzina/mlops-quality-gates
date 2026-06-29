import sys
import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_TEST_SPLIT_PATH = PROJECT_ROOT / "steps" / "data" / "train_test_split.py"

sys.path.append(str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "train_test_split_step_module",
    TRAIN_TEST_SPLIT_PATH,
)
train_test_split_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_test_split_module)


def create_sample_dataframe() -> pd.DataFrame:
    rows = []

    for i in range(100):
        rows.append(
            {
                "duration": 12 + i,
                "credit_amount": 1000 + i,
                "monthly_payment": 100.0,
                "checking_status": "<0",
                "sex": "male" if i % 2 == 0 else "female",
                "class": 0 if i < 70 else 1,
            }
        )

    return pd.DataFrame(rows)


def run_train_test_split(df: pd.DataFrame):
    return train_test_split_module.train_test_split_step.entrypoint(df)


def test_train_test_split_returns_five_outputs():
    df = create_sample_dataframe()

    result = run_train_test_split(df)

    assert len(result) == 5


def test_train_test_split_uses_80_20_split():
    df = create_sample_dataframe()

    X_train, X_test, y_train, y_test, sex_test = run_train_test_split(df)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
    assert len(sex_test) == 20


def test_train_test_split_removes_target_and_sensitive_attribute_from_features():
    df = create_sample_dataframe()

    X_train, X_test, _, _, _ = run_train_test_split(df)

    assert "class" not in X_train.columns
    assert "sex" not in X_train.columns

    assert "class" not in X_test.columns
    assert "sex" not in X_test.columns


def test_train_test_split_keeps_feature_columns():
    df = create_sample_dataframe()

    X_train, X_test, _, _, _ = run_train_test_split(df)

    expected_feature_columns = {
        "duration",
        "credit_amount",
        "monthly_payment",
        "checking_status",
    }

    assert set(X_train.columns) == expected_feature_columns
    assert set(X_test.columns) == expected_feature_columns


def test_train_test_split_preserves_class_distribution_with_stratification():
    df = create_sample_dataframe()

    _, _, y_train, y_test, _ = run_train_test_split(df)

    train_distribution = y_train.value_counts(normalize=True).to_dict()
    test_distribution = y_test.value_counts(normalize=True).to_dict()

    assert train_distribution[0] == 0.7
    assert train_distribution[1] == 0.3

    assert test_distribution[0] == 0.7
    assert test_distribution[1] == 0.3


def test_train_test_split_returns_y_as_series():
    df = create_sample_dataframe()

    _, _, y_train, y_test, _ = run_train_test_split(df)

    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)


def test_train_test_split_returns_sex_test_as_series():
    df = create_sample_dataframe()

    _, _, _, _, sex_test = run_train_test_split(df)

    assert isinstance(sex_test, pd.Series)


def test_train_test_split_sex_test_matches_test_indices():
    df = create_sample_dataframe()

    _, X_test, _, _, sex_test = run_train_test_split(df)

    expected_sex_test = df.loc[X_test.index, "sex"]

    pd.testing.assert_series_equal(
        sex_test.sort_index(),
        expected_sex_test.sort_index(),
        check_names=False,
    )


def test_train_test_split_is_reproducible():
    df = create_sample_dataframe()

    result_1 = run_train_test_split(df)
    result_2 = run_train_test_split(df)

    X_train_1, X_test_1, y_train_1, y_test_1, sex_test_1 = result_1
    X_train_2, X_test_2, y_train_2, y_test_2, sex_test_2 = result_2

    pd.testing.assert_frame_equal(X_train_1, X_train_2)
    pd.testing.assert_frame_equal(X_test_1, X_test_2)
    pd.testing.assert_series_equal(y_train_1, y_train_2)
    pd.testing.assert_series_equal(y_test_1, y_test_2)
    pd.testing.assert_series_equal(sex_test_1, sex_test_2)