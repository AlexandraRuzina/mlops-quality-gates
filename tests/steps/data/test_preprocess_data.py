import sys
import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREPROCESS_DATA_PATH = PROJECT_ROOT / "steps" / "data" / "preprocess_data.py"

sys.path.append(str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "preprocess_data",
    PREPROCESS_DATA_PATH,
)
preprocess_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess_data)


def create_valid_row(**overrides):
    row = {
        "checking_status": "<0",
        "duration": 24,
        "credit_history": "existing paid",
        "purpose": "new car",
        "credit_amount": 3000,
        "savings_status": "<100",
        "employment": "1<=X<4",
        "installment_commitment": 2,
        "personal_status": "male single",
        "other_parties": "none",
        "residence_since": 2,
        "property_magnitude": "car",
        "age": 35,
        "other_payment_plans": "none",
        "housing": "own",
        "existing_credits": 1,
        "job": "skilled",
        "num_dependents": 1,
        "foreign_worker": "yes",
        "class": "good",
    }
    row.update(overrides)
    return row


def run_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return preprocess_data.preprocess_data.entrypoint(df)


def test_preprocess_data_returns_dataframe_for_valid_input():
    df = pd.DataFrame([create_valid_row()])

    result = run_preprocess(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_preprocess_data_removes_rows_with_missing_values():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(credit_amount=None),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert result.loc[0, "credit_amount"] == 3000


def test_preprocess_data_removes_duplicate_rows():
    row = create_valid_row()
    df = pd.DataFrame([row, row.copy()])

    result = run_preprocess(df)

    assert len(result) == 1


def test_preprocess_data_converts_numeric_columns_to_numeric():
    df = pd.DataFrame(
        [
            create_valid_row(
                duration="24",
                credit_amount="3000",
                installment_commitment="2",
                residence_since="2",
                age="35",
                existing_credits="1",
                num_dependents="1",
            )
        ]
    )

    result = run_preprocess(df)

    numeric_columns = [
        "duration",
        "credit_amount",
        "installment_commitment",
        "residence_since",
        "age",
        "existing_credits",
        "num_dependents",
    ]

    for column in numeric_columns:
        assert pd.api.types.is_numeric_dtype(result[column])


def test_preprocess_data_removes_non_numeric_values_in_numeric_columns():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(duration="not_numeric"),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert result.loc[0, "duration"] == 24


def test_preprocess_data_removes_invalid_duration():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(duration=0),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert (result["duration"] >= 1).all()


def test_preprocess_data_removes_invalid_credit_amount():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(credit_amount=99),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert (result["credit_amount"] >= 100).all()


def test_preprocess_data_removes_invalid_installment_commitment():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(installment_commitment=0),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert (result["installment_commitment"] >= 1).all()


def test_preprocess_data_removes_invalid_residence_since():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(residence_since=0),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert (result["residence_since"] >= 1).all()


def test_preprocess_data_removes_invalid_age():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(age=17),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert (result["age"] >= 18).all()


def test_preprocess_data_removes_invalid_existing_credits():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(existing_credits=0),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert (result["existing_credits"] >= 1).all()


def test_preprocess_data_allows_zero_num_dependents():
    df = pd.DataFrame(
        [
            create_valid_row(num_dependents=0),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert result.loc[0, "num_dependents"] == 0


def test_preprocess_data_removes_invalid_checking_status():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(checking_status="invalid"),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert result.loc[0, "checking_status"] == "<0"


def test_preprocess_data_removes_invalid_credit_history():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(credit_history="invalid"),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert result.loc[0, "credit_history"] == "existing paid"


def test_preprocess_data_removes_invalid_purpose():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(purpose="invalid"),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert result.loc[0, "purpose"] == "new car"


def test_preprocess_data_removes_invalid_savings_status():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(savings_status="invalid"),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert result.loc[0, "savings_status"] == "<100"


def test_preprocess_data_removes_invalid_employment():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(employment="invalid"),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert result.loc[0, "employment"] == "1<=X<4"


def test_preprocess_data_removes_invalid_other_parties():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(other_parties="invalid"),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert result.loc[0, "other_parties"] == "none"


def test_preprocess_data_removes_invalid_property_magnitude():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(property_magnitude="invalid"),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert result.loc[0, "property_magnitude"] == "car"


def test_preprocess_data_removes_invalid_other_payment_plans():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(other_payment_plans="invalid"),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert result.loc[0, "other_payment_plans"] == "none"


def test_preprocess_data_removes_invalid_housing():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(housing="invalid"),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert result.loc[0, "housing"] == "own"


def test_preprocess_data_removes_invalid_job():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(job="invalid"),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert result.loc[0, "job"] == "skilled"


def test_preprocess_data_removes_invalid_personal_status():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(personal_status="invalid"),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert result.loc[0, "personal_status"] == "male single"


def test_preprocess_data_removes_invalid_foreign_worker():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(foreign_worker="invalid"),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert result.loc[0, "foreign_worker"] == "yes"


def test_preprocess_data_removes_invalid_target_values():
    df = pd.DataFrame(
        [
            create_valid_row(),
            create_valid_row(**{"class": "invalid"}),
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert result.loc[0, "class"] == "good"


def test_preprocess_data_strips_whitespace_from_categorical_values():
    df = pd.DataFrame(
        [
            create_valid_row(
                checking_status=" <0 ",
                purpose=" new car ",
                housing=" own ",
            )
        ]
    )

    result = run_preprocess(df)

    assert len(result) == 1
    assert result.loc[0, "checking_status"] == "<0"
    assert result.loc[0, "purpose"] == "new car"
    assert result.loc[0, "housing"] == "own"


def test_preprocess_data_resets_index_after_filtering():
    df = pd.DataFrame(
        [
            create_valid_row(duration=0),
            create_valid_row(),
        ]
    )

    result = run_preprocess(df)

    assert result.index.tolist() == [0]