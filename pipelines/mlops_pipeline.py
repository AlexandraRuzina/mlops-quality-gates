from zenml import pipeline

from steps import (
    data_loader,
    data_validation_gate,
    preprocess_data
)

@pipeline
def mlops_pipeline():
    df = data_loader.load_data()
    validated_df = data_validation_gate.data_validation_gate(df)
    preprocess_df = preprocess_data.preprocess_data(validated_df)
