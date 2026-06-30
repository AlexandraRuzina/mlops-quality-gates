import sys
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_PIPELINE_PATH = (
    PROJECT_ROOT / "pipelines" / "experiment_pipeline.py"
)

sys.path.append(str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "experiment_pipeline",
    EXPERIMENT_PIPELINE_PATH,
)
experiment_pipeline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_pipeline_module)


def test_experiment_pipeline_function_exists():
    assert hasattr(experiment_pipeline_module, "experiment_pipeline")


def test_experiment_pipeline_is_callable():
    assert callable(experiment_pipeline_module.experiment_pipeline)


def test_experiment_pipeline_can_be_instantiated():
    pipeline_instance = experiment_pipeline_module.experiment_pipeline

    assert pipeline_instance is not None