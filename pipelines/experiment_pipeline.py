from zenml import pipeline

from steps.experiment import tuning_logistic_regression, tuning_random_forest
from steps.data import data_loader, data_validation_gate, data_post_processing_gate, feature_engineering, \
    preprocess_data, target_encoding, train_test_split


@pipeline
def experiment_pipeline():
    df = data_loader.load_data()
    validated_df = data_validation_gate.data_validation_gate(df)
    preprocess_df = preprocess_data.preprocess_data(validated_df)
    feature_engineering_df, sensitive_attributes = feature_engineering.feature_engineering(preprocess_df)
    post_validated_df = data_post_processing_gate.data_post_processing_gate(feature_engineering_df)
    encoded_target_df = target_encoding.encode_target(post_validated_df)
    X_train, X_test, y_train, y_test = train_test_split.train_test_split_step(encoded_target_df)
    best_params = tuning_random_forest.tuning_random_forest(X_train, y_train)
    best_lr_params = tuning_logistic_regression.tuning_logistic_regression(X_train, y_train)

if __name__ == "__main__":
    experiment_pipeline()