import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CREATE_QUALITY_REPORT_PATH = (
    PROJECT_ROOT / "steps" / "create_quality_report.py"
)

sys.path.append(str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "create_quality_report",
    CREATE_QUALITY_REPORT_PATH,
)
create_quality_report_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(create_quality_report_module)


class DummyPipelineRun:
    id = "dummy-zenml-run-id"
    name = "dummy-pipeline-run"


class DummyContext:
    pipeline_run = DummyPipelineRun()


class DummyMLflowRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def patch_mlflow(monkeypatch):
    calls = {
        "tracking_uri": None,
        "run_id": None,
        "tags": {},
        "params": {},
        "metrics": {},
        "artifacts": [],
    }

    monkeypatch.setattr(
        create_quality_report_module.mlflow,
        "set_tracking_uri",
        lambda uri: calls.update({"tracking_uri": uri}),
    )

    def fake_start_run(run_id):
        calls["run_id"] = run_id
        return DummyMLflowRun()

    monkeypatch.setattr(
        create_quality_report_module.mlflow,
        "start_run",
        fake_start_run,
    )

    monkeypatch.setattr(
        create_quality_report_module.mlflow,
        "set_tag",
        lambda key, value: calls["tags"].update({key: value}),
    )

    monkeypatch.setattr(
        create_quality_report_module.mlflow,
        "log_param",
        lambda key, value: calls["params"].update({key: value}),
    )

    monkeypatch.setattr(
        create_quality_report_module.mlflow,
        "log_metric",
        lambda key, value: calls["metrics"].update({key: value}),
    )

    monkeypatch.setattr(
        create_quality_report_module.mlflow,
        "log_artifact",
        lambda path: calls["artifacts"].append(path),
    )

    return calls


@pytest.fixture
def sample_reports():
    performance_report = {
        "gate_name": "Performance Quality Gate",
        "gate_passed": True,
        "metrics": {
            "accuracy": {
                "value": 0.75,
                "threshold": 0.70,
                "comparison": ">",
                "passed": True,
            }
        },
    }

    fairness_report = {
        "gate_name": "Fairness Quality Gate",
        "gate_passed": True,
        "metrics": {
            "separation_difference": {
                "value": 0.03,
                "threshold": 0.05,
                "comparison": "<=",
                "passed": np.bool_(True),
            }
        },
    }

    robustness_report = {
        "gate_name": "Robustness Quality Gate",
        "gate_passed": True,
        "metrics": {
            "mean_robustness_score": {
                "value": 0.97,
                "threshold": 0.95,
                "comparison": ">=",
                "passed": True,
            }
        },
    }

    metamorphic_report = {
        "gate_name": "Metamorphic Verification Gate",
        "gate_passed": True,
        "metrics": {
            "duration_increase": {
                "number_of_cases": np.int64(100),
                "number_of_violations": np.int64(2),
                "violation_rate": np.float64(0.02),
                "threshold": 0.10,
                "comparison": "<=",
                "passed": np.bool_(True),
            }
        },
    }

    return {
        "performance_report": performance_report,
        "fairness_report": fairness_report,
        "robustness_report": robustness_report,
        "metamorphic_report": metamorphic_report,
    }


def run_create_quality_report(
    reports,
    performance_gate_passed=True,
    fairness_gate_passed=True,
    robustness_gate_passed=True,
    metamorphic_gate_passed=True,
    mlflow_run_id="dummy-mlflow-run-id",
):
    return create_quality_report_module.create_quality_report.entrypoint(
        reports["performance_report"],
        performance_gate_passed,
        reports["fairness_report"],
        fairness_gate_passed,
        reports["robustness_report"],
        robustness_gate_passed,
        reports["metamorphic_report"],
        metamorphic_gate_passed,
        mlflow_run_id,
    )


def test_create_quality_report_returns_report_and_boolean(
    monkeypatch,
    tmp_path,
    sample_reports,
):
    monkeypatch.chdir(tmp_path)
    patch_mlflow(monkeypatch)

    monkeypatch.setattr(
        create_quality_report_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    quality_report, production_readiness_passed = run_create_quality_report(
        sample_reports
    )

    assert isinstance(quality_report, dict)
    assert isinstance(production_readiness_passed, bool)


def test_create_quality_report_passes_when_all_gates_pass(
    monkeypatch,
    tmp_path,
    sample_reports,
):
    monkeypatch.chdir(tmp_path)
    patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        create_quality_report_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    quality_report, production_readiness_passed = run_create_quality_report(
        sample_reports
    )

    assert production_readiness_passed is True
    assert quality_report["production_readiness_passed"] is True


def test_create_quality_report_fails_when_one_gate_fails(
    monkeypatch,
    tmp_path,
    sample_reports,
):
    monkeypatch.chdir(tmp_path)
    patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        create_quality_report_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    quality_report, production_readiness_passed = run_create_quality_report(
        sample_reports,
        fairness_gate_passed=False,
    )

    assert production_readiness_passed is False
    assert quality_report["production_readiness_passed"] is False


def test_create_quality_report_contains_all_gate_sections(
    monkeypatch,
    tmp_path,
    sample_reports,
):
    monkeypatch.chdir(tmp_path)
    patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        create_quality_report_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    quality_report, _ = run_create_quality_report(sample_reports)

    expected_gates = {
        "data_validation_gate",
        "data_post_processing_gate",
        "performance_gate",
        "fairness_gate",
        "robustness_gate",
        "metamorphic_verification_gate",
    }

    assert set(quality_report["gates"].keys()) == expected_gates


def test_create_quality_report_creates_html_and_json_files(
    monkeypatch,
    tmp_path,
    sample_reports,
):
    monkeypatch.chdir(tmp_path)
    patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        create_quality_report_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    _ = run_create_quality_report(sample_reports)

    html_path = tmp_path / "reports" / "production_readiness_quality_report.html"
    json_path = tmp_path / "reports" / "production_readiness_quality_report.json"

    assert html_path.exists()
    assert json_path.exists()


def test_create_quality_report_html_contains_gate_names(
    monkeypatch,
    tmp_path,
    sample_reports,
):
    monkeypatch.chdir(tmp_path)
    patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        create_quality_report_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    _ = run_create_quality_report(sample_reports)

    html_path = tmp_path / "reports" / "production_readiness_quality_report.html"
    html_content = html_path.read_text(encoding="utf-8")

    assert "Production Readiness Quality Report" in html_content
    assert "Performance Quality Gate" in html_content
    assert "Fairness Quality Gate" in html_content
    assert "Robustness Quality Gate" in html_content
    assert "Metamorphic Verification Gate" in html_content


def test_create_quality_report_json_is_valid_and_serializable(
    monkeypatch,
    tmp_path,
    sample_reports,
):
    monkeypatch.chdir(tmp_path)
    patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        create_quality_report_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    _ = run_create_quality_report(sample_reports)

    json_path = tmp_path / "reports" / "production_readiness_quality_report.json"

    with open(json_path, "r", encoding="utf-8") as file:
        json_report = json.load(file)

    assert json_report["report_name"] == "Production Readiness Quality Report"
    assert isinstance(
        json_report["gates"]["fairness_gate"]["metrics"]
        ["separation_difference"]["passed"],
        bool,
    )
    assert isinstance(
        json_report["gates"]["metamorphic_verification_gate"]["metrics"]
        ["duration_increase"]["number_of_cases"],
        int,
    )


def test_create_quality_report_logs_to_existing_mlflow_run(
    monkeypatch,
    tmp_path,
    sample_reports,
):
    monkeypatch.chdir(tmp_path)
    calls = patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        create_quality_report_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    _ = run_create_quality_report(
        sample_reports,
        mlflow_run_id="existing-run-id",
    )

    assert calls["tracking_uri"] == "http://127.0.0.1:5000"
    assert calls["run_id"] == "existing-run-id"


def test_create_quality_report_logs_zenml_tags(
    monkeypatch,
    tmp_path,
    sample_reports,
):
    monkeypatch.chdir(tmp_path)
    calls = patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        create_quality_report_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    _ = run_create_quality_report(sample_reports)

    assert calls["tags"]["zenml_run_id"] == "dummy-zenml-run-id"
    assert calls["tags"]["zenml_pipeline_run_name"] == "dummy-pipeline-run"


def test_create_quality_report_logs_gate_metrics_to_mlflow(
    monkeypatch,
    tmp_path,
    sample_reports,
):
    monkeypatch.chdir(tmp_path)
    calls = patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        create_quality_report_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    _ = run_create_quality_report(
        sample_reports,
        fairness_gate_passed=False,
    )

    assert calls["params"]["production_readiness_passed"] is False
    assert calls["metrics"]["performance_gate_passed"] == 1
    assert calls["metrics"]["fairness_gate_passed"] == 0
    assert calls["metrics"]["robustness_gate_passed"] == 1
    assert calls["metrics"]["metamorphic_gate_passed"] == 1


def test_create_quality_report_logs_html_and_json_artifacts(
    monkeypatch,
    tmp_path,
    sample_reports,
):
    monkeypatch.chdir(tmp_path)
    calls = patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        create_quality_report_module,
        "get_step_context",
        lambda: DummyContext(),
    )

    _ = run_create_quality_report(sample_reports)

    artifact_names = [Path(path).name for path in calls["artifacts"]]

    assert "production_readiness_quality_report.html" in artifact_names
    assert "production_readiness_quality_report.json" in artifact_names
    assert len(calls["artifacts"]) == 2