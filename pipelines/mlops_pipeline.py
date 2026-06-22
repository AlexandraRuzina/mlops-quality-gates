from zenml import pipeline

from steps import (
    data_loader,
    data_validation_gate
)

@pipeline
def mlops_pipeline():
    df = data_loader.load_data()
    validated_df = data_validation_gate.data_validation_gate(df)
