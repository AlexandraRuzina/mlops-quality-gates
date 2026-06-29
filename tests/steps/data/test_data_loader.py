import sys
import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_LOADER_PATH = PROJECT_ROOT / "steps" / "data" / "data_loader.py"

sys.path.append(str(PROJECT_ROOT))


spec = importlib.util.spec_from_file_location(
    "data_loader",
    DATA_LOADER_PATH,
)
data_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_loader)


def test_load_data_returns_dataframe(monkeypatch):
    expected_df = pd.DataFrame(
        {
            "duration": [12, 24],
            "credit_amount": [1000, 2000],
            "class": ["good", "bad"],
        }
    )

    monkeypatch.setattr(
        data_loader.pd,
        "read_parquet",
        lambda path: expected_df,
    )

    result = data_loader.load_data.entrypoint()

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected_df)


def test_load_data_uses_expected_parquet_path(monkeypatch):
    captured_path = {}

    expected_df = pd.DataFrame(
        {
            "duration": [12],
            "credit_amount": [1000],
            "class": ["good"],
        }
    )

    def fake_read_parquet(path):
        captured_path["path"] = path
        return expected_df

    monkeypatch.setattr(
        data_loader.pd,
        "read_parquet",
        fake_read_parquet,
    )

    _ = data_loader.load_data.entrypoint()

    assert captured_path["path"].name == "german_credit.parquet"
    assert captured_path["path"].parent.name == "raw"
    assert captured_path["path"].parent.parent.name == "data"


def test_load_data_preserves_shape(monkeypatch):
    expected_df = pd.DataFrame(
        {
            "checking_status": ["<0", ">=200", "no checking"],
            "duration": [12, 24, 36],
            "credit_amount": [1000, 2000, 3000],
            "class": ["good", "bad", "good"],
        }
    )

    monkeypatch.setattr(
        data_loader.pd,
        "read_parquet",
        lambda path: expected_df,
    )

    result = data_loader.load_data.entrypoint()

    assert result.shape == (3, 4)