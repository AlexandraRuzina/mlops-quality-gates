import sys
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_READINESS_PIPELINE_PATH = (
    PROJECT_ROOT / "pipelines" / "production_readiness_pipeline.py"
)

sys.path.append(str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "production_readiness_pipeline",
    PRODUCTION_READINESS_PIPELINE_PATH,
)
production_readiness_pipeline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production_readiness_pipeline_module)


def test_production_readiness_pipeline_function_exists():
    assert hasattr(
        production_readiness_pipeline_module,
        "production_readiness_pipeline",
    )


def test_production_readiness_pipeline_is_callable():
    assert callable(
        production_readiness_pipeline_module.production_readiness_pipeline
    )


def test_production_readiness_pipeline_can_be_referenced():
    pipeline_instance = (
        production_readiness_pipeline_module.production_readiness_pipeline
    )

    assert pipeline_instance is not None


def test_production_readiness_pipeline_source_contains_expected_steps():
    source_code = Path(PRODUCTION_READINESS_PIPELINE_PATH).read_text(
        encoding="utf-8"
    )

    expected_step_calls = [
        "data_loader.load_data",
        "data_validation_gate.data_validation_gate",
        "preprocess_data.preprocess_data",
        "feature_engineering.feature_engineering",
        "target_encoding.encode_target",
        "data_post_processing_gate.data_post_processing_gate",
        "train_test_split.train_test_split_step",
        "final_training_random_forest.train_final_random_forest",
        "calculate_dummy_metrics.evaluate_dummy_baseline",
        "test_evaluation.test_evaluation",
        "performance_gate.performance_gate",
        "fairness_gate.fairness_gate",
        "robustness_gate.robustness_gate",
        "metamorphic_verification_gate.metamorphic_verification_gate",
        "create_quality_report.create_quality_report",
        "production_readiness_gate.production_readiness_gate",
        "register_model.register_model",
    ]

    for step_call in expected_step_calls:
        assert step_call in source_code


def test_production_readiness_pipeline_has_cache_disabled():
    source_code = Path(PRODUCTION_READINESS_PIPELINE_PATH).read_text(
        encoding="utf-8"
    )

    assert "@pipeline(enable_cache=False)" in source_code


def test_production_readiness_pipeline_registers_model_after_readiness_gate():
    source_code = Path(PRODUCTION_READINESS_PIPELINE_PATH).read_text(
        encoding="utf-8"
    )

    readiness_gate_position = source_code.find(
        "production_readiness_gate.production_readiness_gate"
    )
    register_model_position = source_code.find(
        "register_model.register_model"
    )

    assert readiness_gate_position != -1
    assert register_model_position != -1
    assert readiness_gate_position < register_model_position


def test_production_readiness_pipeline_passes_mlflow_run_id_to_report_and_registry():
    source_code = Path(PRODUCTION_READINESS_PIPELINE_PATH).read_text(
        encoding="utf-8"
    )

    assert "mlflow_run_id" in source_code
    assert "create_quality_report.create_quality_report" in source_code
    assert "register_model.register_model(production_readiness_checked, mlflow_run_id)" in source_code