import sys
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(APP_DIR))


spec = importlib.util.spec_from_file_location(
    "streamlit_app",
    APP_DIR / "app.py",
)
streamlit_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(streamlit_app)


def mock_streamlit_inputs(monkeypatch):
    monkeypatch.setattr(streamlit_app.st, "subheader", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        streamlit_app.st,
        "number_input",
        lambda label, **kwargs: kwargs.get("value"),
    )

    monkeypatch.setattr(
        streamlit_app.st,
        "selectbox",
        lambda label, options, **kwargs: options[kwargs.get("index", 0)],
    )

    monkeypatch.setattr(streamlit_app.st, "error", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "stop", lambda: (_ for _ in ()).throw(RuntimeError("st.stop called")))


def test_create_input_dataframe_returns_dataframe(monkeypatch):
    mock_streamlit_inputs(monkeypatch)

    input_df = streamlit_app.create_input_dataframe()

    assert isinstance(input_df, pd.DataFrame)
    assert len(input_df) == 1

    expected_columns = {
        "checking_status",
        "duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings_status",
        "employment",
        "installment_commitment",
        "other_parties",
        "residence_since",
        "property_magnitude",
        "age",
        "other_payment_plans",
        "housing",
        "existing_credits",
        "job",
        "num_dependents",
    }

    assert set(input_df.columns) == expected_columns


def test_create_input_dataframe_numeric_columns_are_numeric(monkeypatch):
    mock_streamlit_inputs(monkeypatch)

    input_df = streamlit_app.create_input_dataframe()

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
        assert pd.api.types.is_numeric_dtype(input_df[column])


def test_create_input_dataframe_stops_if_field_is_missing(monkeypatch):
    monkeypatch.setattr(streamlit_app.st, "subheader", lambda *args, **kwargs: None)

    def fake_number_input(label, **kwargs):
        if label == "Age":
            return None
        return kwargs.get("value")

    monkeypatch.setattr(streamlit_app.st, "number_input", fake_number_input)

    monkeypatch.setattr(
        streamlit_app.st,
        "selectbox",
        lambda label, options, **kwargs: options[kwargs.get("index", 0)],
    )

    monkeypatch.setattr(streamlit_app.st, "error", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "stop", lambda: (_ for _ in ()).throw(RuntimeError("st.stop called")))

    with pytest.raises(RuntimeError, match="st.stop called"):
        streamlit_app.create_input_dataframe()


def test_create_input_dataframe_stops_if_numeric_field_is_not_numeric(monkeypatch):
    monkeypatch.setattr(streamlit_app.st, "subheader", lambda *args, **kwargs: None)

    def fake_number_input(label, **kwargs):
        if label == "Duration (months)":
            return "not_numeric"
        return kwargs.get("value")

    monkeypatch.setattr(streamlit_app.st, "number_input", fake_number_input)

    monkeypatch.setattr(
        streamlit_app.st,
        "selectbox",
        lambda label, options, **kwargs: options[kwargs.get("index", 0)],
    )

    monkeypatch.setattr(streamlit_app.st, "error", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "stop", lambda: (_ for _ in ()).throw(RuntimeError("st.stop called")))

    with pytest.raises(RuntimeError, match="st.stop called"):
        streamlit_app.create_input_dataframe()


def test_main_calls_prediction_with_feature_engineered_data(monkeypatch):
    raw_input_df = pd.DataFrame(
        [{
            "checking_status": "<0",
            "duration": 24,
            "credit_history": "existing paid",
            "purpose": "new car",
            "credit_amount": 3000,
            "savings_status": "<100",
            "employment": "1<=X<4",
            "installment_commitment": 2,
            "other_parties": "none",
            "residence_since": 2,
            "property_magnitude": "car",
            "age": 35,
            "other_payment_plans": "none",
            "housing": "own",
            "existing_credits": 1,
            "job": "skilled",
            "num_dependents": 1,
        }]
    )

    feature_df = raw_input_df.copy()
    feature_df["monthly_payment"] = 125.0
    feature_df["credit_burden_ratio"] = 2 / 3000
    feature_df["age_group"] = "25-40"
    feature_df["is_negative_checking"] = True
    feature_df["has_guarantor"] = False
    feature_df["has_additional_security"] = False

    captured_input = {}

    monkeypatch.setattr(streamlit_app.st, "set_page_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "title", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "button", lambda *args, **kwargs: True)
    monkeypatch.setattr(streamlit_app.st, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "error", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "dataframe", lambda *args, **kwargs: None)

    class DummyExpander:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(streamlit_app.st, "expander", lambda *args, **kwargs: DummyExpander())

    monkeypatch.setattr(streamlit_app, "create_input_dataframe", lambda: raw_input_df)
    monkeypatch.setattr(streamlit_app.inference, "load_model", lambda: "dummy_model")
    monkeypatch.setattr(
        streamlit_app.calculate_features,
        "calculate_features",
        lambda df: feature_df,
    )

    def fake_predict_credit_risk(model, input_data):
        captured_input["model"] = model
        captured_input["input_data"] = input_data
        return {
            "prediction": 0,
            "prediction_label": "good",
            "probability_good": 0.8,
            "probability_bad": 0.2,
        }

    monkeypatch.setattr(
        streamlit_app.inference,
        "predict_credit_risk",
        fake_predict_credit_risk,
    )

    streamlit_app.main()

    assert captured_input["model"] == "dummy_model"
    assert "age" not in captured_input["input_data"].columns
    assert "monthly_payment" in captured_input["input_data"].columns