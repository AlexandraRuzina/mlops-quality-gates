from zenml import pipeline

from steps import (
    data_loader,
    data_validation_gate,
    preprocess_data,
    feature_engineering,
    data_post_processing_gate
)

@pipeline
def mlops_pipeline():
    df = data_loader.load_data()
    validated_df = data_validation_gate.data_validation_gate(df)
    preprocess_df = preprocess_data.preprocess_data(validated_df)
    feature_engineering_df = feature_engineering.feature_engineering(preprocess_df)
    post_validated_df = data_post_processing_gate.data_post_processing_gate(feature_engineering_df)
