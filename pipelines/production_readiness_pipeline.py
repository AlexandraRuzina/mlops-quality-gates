from zenml import pipeline
from zenml.client import Client
from steps.data import data_loader, data_validation_gate, data_post_processing_gate, feature_engineering, \
    preprocess_data, target_encoding, train_test_split
from steps.finalCheck import final_training_random_forest, test_evaluation, calculate_dummy_metrics, performance_gate, \
    fairness_gate, robustness_gate, metamorphic_verification_gate
from steps import register_model


@pipeline
def production_readiness_pipeline(best_rf_params: dict):
    df = data_loader.load_data()
    validated_df = data_validation_gate.data_validation_gate(df)
    preprocess_df = preprocess_data.preprocess_data(validated_df)
    feature_engineering_df = feature_engineering.feature_engineering(preprocess_df)
    encoded_target_df = target_encoding.encode_target(feature_engineering_df)
    post_validated_df = data_post_processing_gate.data_post_processing_gate(encoded_target_df)
    X_train, X_test, y_train, y_test, sex_test = train_test_split.train_test_split_step(post_validated_df)
    rf_pipeline = final_training_random_forest.train_final_random_forest(X_train, y_train, best_rf_params)
    dummy_accuracy = calculate_dummy_metrics.evaluate_dummy_baseline(X_train, X_test, y_train, y_test)
    rf_test_metrics, y_pred, y_proba = test_evaluation.test_evaluation(
        model_pipeline=rf_pipeline,
        X_test=X_test,
        y_test=y_test,
        model_name="random_forest",
    )
    performance_gate.performance_gate(rf_test_metrics, dummy_accuracy)
    fairness_gate.fairness_gate(y_test, y_pred, sex_test)
    robustness_metrics = robustness_gate.robustness_gate(rf_pipeline, X_test, y_test, rf_test_metrics)
    metamorphic_metrics = metamorphic_verification_gate.metamorphic_verification_gate(rf_pipeline,X_test,y_proba)
    registered_model_data = register_model.register_model(rf_pipeline)



if __name__ == "__main__":
    client = Client()

    artifact_rf = client.get_artifact_version(name_id_or_prefix="best_random_forest_params")
    params_rf = artifact_rf.load()

    production_readiness_pipeline(params_rf)