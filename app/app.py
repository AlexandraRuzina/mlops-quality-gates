import pandas as pd
import streamlit as st

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import scripts.calculate_features as calculate_features
import inference



def create_input_dataframe() -> pd.DataFrame:
    st.subheader("Credit Information")

    duration = st.number_input(
        "Duration (months)",
        min_value=1,
        value=24,
        step=1,
        help="Duration of the requested loan in months. Example: 24 means a loan term of 24 months.",
    )

    credit_amount = st.number_input(
        "Credit Amount",
        min_value=1,
        value=3000,
        step=100,
        help="Requested loan amount. Enter a positive numeric value.",
    )

    purpose = st.selectbox(
        "Purpose",
        [
            "new car",
            "used car",
            "furniture/equipment",
            "radio/tv",
            "domestic appliance",
            "repairs",
            "education",
            "retraining",
            "business",
            "other",
        ],
        help=(
            "Purpose of the loan. For example, 'new car' means the loan is used "
            "to buy a new car, while 'business' means it is used for business purposes."
        ),
    )

    installment_commitment = st.number_input(
        "Installment Commitment",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
        help=(
            "Installment rate as a percentage category of disposable income. "
            "Allowed values are 1 to 4. Higher values indicate a higher monthly burden."
        ),
    )

    st.subheader("Applicant Information")

    age = st.number_input(
        "Age",
        min_value=18,
        value=35,
        step=1,
        help="Age of the applicant in years. The value must be at least 18.",
    )

    employment = st.selectbox(
        "Employment",
        ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
        help=(
            "Employment duration of the applicant. '<1' means less than one year, "
            "'1<=X<4' means between one and four years, '4<=X<7' means between "
            "four and seven years, and '>=7' means seven years or more."
        ),
    )

    job = st.selectbox(
        "Job",
        [
            "unemp/unskilled non res",
            "unskilled resident",
            "skilled",
            "high qualif/self emp/mgmt",
        ],
        help=(
            "Job category of the applicant. 'unemp/unskilled non res' means unemployed "
            "or unskilled non-resident, 'unskilled resident' means unskilled resident, "
            "'skilled' means skilled employee, and 'high qualif/self emp/mgmt' means "
            "highly qualified, self-employed or management position."
        ),
    )

    residence_since = st.number_input(
        "Residence Since",
        min_value=1,
        value=2,
        step=1,
        help=(
            "Length of residence in the current home, given in years."
        ),
    )

    num_dependents = st.number_input("Number of Dependents",
        min_value=0,
        value=1,
        step=1,
        help="Number of people financially dependent on the applicant.")

    st.subheader("Financial Situation")

    checking_status = st.selectbox(
        "Checking Status",
        ["<0", "0<=X<200", ">=200", "no checking"],
        help=(
            "Status of the applicant's checking account. '<0' means negative balance, "
            "'0<=X<200' means balance between 0 and 200, '>=200' means balance of at "
            "least 200, and 'no checking' means no checking account is available."
        ),
    )

    savings_status = st.selectbox(
        "Savings Status",
        ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
        help=(
            "Savings account status. '<100' means less than 100, '100<=X<500' means "
            "between 100 and 500, '500<=X<1000' means between 500 and 1000, "
            "'>=1000' means at least 1000, and 'no known savings' means no known savings."
        ),
    )

    existing_credits = st.number_input(
        "Existing Credits",
        min_value=1,
        value=1,
        step=1,
        help="Number of existing credits at the bank. Enter a positive integer value.",
    )

    credit_history = st.selectbox(
        "Credit History",
        [
            "no credits/all paid",
            "all paid",
            "existing paid",
            "delayed previously",
            "critical/other existing credit",
        ],
        help=(
            "Credit history of the applicant. 'no credits/all paid' means no previous "
            "credits or all credits paid back, 'all paid' means all previous credits paid, "
            "'existing paid' means existing credits have been paid so far, "
            "'delayed previously' means there were previous payment delays, and "
            "'critical/other existing credit' indicates critical credit history or other existing credit."
        ),
    )

    other_payment_plans = st.selectbox(
        "Other Payment Plans",
        ["bank", "stores", "none"],
        help=(
            "Other installment plans outside this credit. 'bank' means another bank plan, "
            "'stores' means store installment plan, and 'none' means no other payment plan."
        ),
    )

    st.subheader("Assets and Housing")

    property_magnitude = st.selectbox(
        "Property",
        ["real estate", "life insurance", "car", "no known property"],
        help=(
            "Type of property or asset owned by the applicant. Options are real estate, "
            "life insurance, car, or no known property."
        ),
    )

    housing = st.selectbox(
        "Housing",
        ["rent", "own", "for free"],
        help=(
            "Housing situation of the applicant. 'rent' means rented housing, "
            "'own' means owned housing, and 'for free' means living rent-free."
        ),
    )

    other_parties = st.selectbox(
        "Other Parties",
        ["none", "co applicant", "guarantor"],
        help=(
            "Other parties involved in the credit application. 'none' means no additional party, "
            "'co applicant' means there is a co-applicant, and 'guarantor' means there is a guarantor."
        ),
    )

    input_data = pd.DataFrame(
        [{
            "checking_status": checking_status,
            "duration": duration,
            "credit_history": credit_history,
            "purpose": purpose,
            "credit_amount": credit_amount,
            "savings_status": savings_status,
            "employment": employment,
            "installment_commitment": installment_commitment,
            "other_parties": other_parties,
            "residence_since": residence_since,
            "property_magnitude": property_magnitude,
            "age": age,
            "other_payment_plans": other_payment_plans,
            "housing": housing,
            "existing_credits": existing_credits,
            "job": job,
            "num_dependents": num_dependents,
        }]
    )

    if input_data.isnull().any().any():
        st.error("All fields must be filled before making a prediction.")
        st.stop()

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
        if not pd.api.types.is_numeric_dtype(input_data[column]):
            st.error(f"The field '{column}' must contain a numeric value.")
            st.stop()

    return input_data


def main():
    st.set_page_config(
        page_title="Credit Risk Prediction",
        page_icon="💳",
        layout="centered",
    )

    st.title("Credit Risk Prediction")
    st.write(
        "This application uses the registered MLflow model to predict "
        "whether a credit application is classified as good or bad credit risk."
    )

    model = inference.load_model()

    st.subheader("Applicant and Credit Information")

    input_df = create_input_dataframe()
    features_df = calculate_features.calculate_features(input_df)
    features_df = features_df.drop(columns=["age"])

    if st.button("Predict Credit Risk"):

        result = inference.predict_credit_risk(
            model=model,
            input_data=features_df,
        )

        st.subheader("Prediction Result")

        if result["prediction"] == 1:
            st.error("Prediction: Bad Credit Risk")
        else:
            st.success("Prediction: Good Credit Risk")

        if (
                "probability_good" in result
                and "probability_bad" in result
        ):
            st.write(
                f"Probability Good: {result['probability_good']:.2%}"
            )
            st.write(
                f"Probability Bad: {result['probability_bad']:.2%}"
            )

        with st.expander("Show processed input data"):
            st.dataframe(features_df)


if __name__ == "__main__":
    main()