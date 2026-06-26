from zenml import pipeline
from zenml.client import Client
from steps.data import data_loader, data_validation_gate, data_post_processing_gate, feature_engineering, \
    preprocess_data, target_encoding, train_test_split
from steps.finalCheck import final_training_random_forest, test_evaluation, calculate_dummy_metrics, performance_gate, fairness_gate


@pipeline
def production_readiness_pipeline(best_rf_params: dict):
    df = data_loader.load_data()
    validated_df = data_validation_gate.data_validation_gate(df)
    preprocess_df = preprocess_data.preprocess_data(validated_df)
    feature_engineering_df = feature_engineering.feature_engineering(preprocess_df)
    post_validated_df = data_post_processing_gate.data_post_processing_gate(feature_engineering_df)
    encoded_target_df = target_encoding.encode_target(post_validated_df)
    X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split.train_test_split_step(encoded_target_df)
    rf_pipeline = final_training_random_forest.train_final_random_forest(X_train, y_train, best_rf_params)
    rf_test_metrics, y_pred =test_evaluation.test_evaluation(
        model_pipeline=rf_pipeline,
        X_test=X_test,
        y_test=y_test,
        model_name="random_forest",
    )
    dummy_accuracy = calculate_dummy_metrics.evaluate_dummy_baseline(X_train, X_test, y_train, y_test)
    performance_gate.performance_gate(rf_test_metrics, dummy_accuracy)
    fairness_gate.fairness_gate(y_test, y_pred, sex_test)


if __name__ == "__main__":
    client = Client()

    artifact_rf = client.get_artifact_version(name_id_or_prefix="best_random_forest_params")
    params_rf = artifact_rf.load()

    production_readiness_pipeline(params_rf)