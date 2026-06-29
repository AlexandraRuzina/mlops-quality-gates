from zenml import step
from typing import Annotated
from datetime import datetime
import os
import json
import mlflow
from zenml import get_step_context


@step
def create_quality_report(
    performance_report: dict,
    performance_gate_passed: bool,
    fairness_report: dict,
    fairness_gate_passed: bool,
    robustness_report: dict,
    robustness_gate_passed: bool,
    metamorphic_report: dict,
    metamorphic_gate_passed: bool,
    mlflow_run_id: str,
) -> tuple[
    Annotated[dict, "quality_report"],
    Annotated[bool, "production_readiness_passed"],
]:
    """
    Creates an HTML quality report containing all quality gate results
    and logs the report as an MLflow artifact.

    The report is logged to the same MLflow run that was created during
    final model training.
    """

    data_validation_gate_passed = True
    data_post_processing_gate_passed = True

    production_readiness_passed = all([
        data_validation_gate_passed,
        data_post_processing_gate_passed,
        performance_gate_passed,
        fairness_gate_passed,
        robustness_gate_passed,
        metamorphic_gate_passed,
    ])

    quality_report = {
        "report_name": "Production Readiness Quality Report",
        "created_at": datetime.now().isoformat(),
        "production_readiness_passed": production_readiness_passed,
        "gates": {
            "data_validation_gate": {
                "gate_name": "Data Validation Gate",
                "gate_passed": data_validation_gate_passed,
            },
            "data_post_processing_gate": {
                "gate_name": "Data Post-Processing Gate",
                "gate_passed": data_post_processing_gate_passed,
            },
            "performance_gate": performance_report,
            "fairness_gate": fairness_report,
            "robustness_gate": robustness_report,
            "metamorphic_verification_gate": metamorphic_report,
        },
    }

    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {
                key: make_json_serializable(value)
                for key, value in obj.items()
            }
        if isinstance(obj, list):
            return [make_json_serializable(value) for value in obj]
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    def status_badge(passed: bool) -> str:
        return (
            "<span style='color: green; font-weight: bold;'>PASSED</span>"
            if passed
            else "<span style='color: red; font-weight: bold;'>FAILED</span>"
        )

    def render_metrics(metrics: dict) -> str:
        rows = ""

        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict) and "value" in metric_data:
                value = metric_data.get("value")
                threshold = metric_data.get("threshold", "-")
                comparison = metric_data.get("comparison", "-")
                passed = metric_data.get("passed", None)

                rows += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{value:.4f}</td>
                    <td>{comparison} {threshold:.4f}</td>
                    <td>{status_badge(passed)}</td>
                </tr>
                """

            elif isinstance(metric_data, dict) and "violation_rate" in metric_data:
                violation_rate = metric_data.get("violation_rate")
                threshold = metric_data.get("threshold", "-")
                comparison = metric_data.get("comparison", "-")
                passed = metric_data.get("passed", None)
                violations = metric_data.get("number_of_violations", "-")
                cases = metric_data.get("number_of_cases", "-")

                rows += f"""
                <tr>
                    <td>{metric_name} ({violations}/{cases} violations)</td>
                    <td>{violation_rate:.4f}</td>
                    <td>{comparison} {threshold:.4f}</td>
                    <td>{status_badge(passed)}</td>
                </tr>
                """

            elif isinstance(metric_data, (int, float)):
                rows += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{metric_data:.4f}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                """

        return rows

    def render_gate_section(gate_key: str, gate_report: dict) -> str:
        gate_name = gate_report.get("gate_name", gate_key)
        gate_passed = gate_report.get("gate_passed", False)
        metrics = gate_report.get("metrics", {})

        metrics_table = ""

        if metrics:
            metrics_table = f"""
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Threshold</th>
                    <th>Status</th>
                </tr>
                {render_metrics(metrics)}
            </table>
            """

        return f"""
        <section>
            <h2>{gate_name}</h2>
            <p><strong>Status:</strong> {status_badge(gate_passed)}</p>
            {metrics_table}
        </section>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Production Readiness Quality Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                color: #222;
            }}
            h1 {{
                border-bottom: 2px solid #333;
                padding-bottom: 10px;
            }}
            h2 {{
                margin-top: 35px;
                color: #333;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 10px;
                margin-bottom: 25px;
            }}
            th, td {{
                border: 1px solid #ccc;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .summary {{
                padding: 15px;
                background-color: #f7f7f7;
                border: 1px solid #ddd;
                margin-bottom: 30px;
            }}
        </style>
    </head>
    <body>
        <h1>Production Readiness Quality Report</h1>

        <div class="summary">
            <p><strong>Created at:</strong> {quality_report["created_at"]}</p>
            <p><strong>Overall Production Readiness:</strong> {status_badge(production_readiness_passed)}</p>
        </div>

        <h2>Gate Overview</h2>
        <table>
            <tr>
                <th>Quality Gate</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>Data Validation Gate</td>
                <td>{status_badge(data_validation_gate_passed)}</td>
            </tr>
            <tr>
                <td>Data Post-Processing Gate</td>
                <td>{status_badge(data_post_processing_gate_passed)}</td>
            </tr>
            <tr>
                <td>Performance Quality Gate</td>
                <td>{status_badge(performance_gate_passed)}</td>
            </tr>
            <tr>
                <td>Fairness Quality Gate</td>
                <td>{status_badge(fairness_gate_passed)}</td>
            </tr>
            <tr>
                <td>Robustness Quality Gate</td>
                <td>{status_badge(robustness_gate_passed)}</td>
            </tr>
            <tr>
                <td>Metamorphic Verification Gate</td>
                <td>{status_badge(metamorphic_gate_passed)}</td>
            </tr>
        </table>

        {render_gate_section("performance_gate", performance_report)}
        {render_gate_section("fairness_gate", fairness_report)}
        {render_gate_section("robustness_gate", robustness_report)}
        {render_gate_section("metamorphic_verification_gate", metamorphic_report)}

    </body>
    </html>
    """

    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)

    html_report_path = os.path.join(
        report_dir,
        "production_readiness_quality_report.html",
    )

    json_report_path = os.path.join(
        report_dir,
        "production_readiness_quality_report.json",
    )

    with open(html_report_path, "w", encoding="utf-8") as file:
        file.write(html_content)

    quality_report = make_json_serializable(quality_report)
    with open(json_report_path, "w", encoding="utf-8") as file:
        json.dump(quality_report, file, indent=4)

    context = get_step_context()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.set_tag(
            "zenml_run_id",
            str(context.pipeline_run.id),
        )

        mlflow.set_tag(
            "zenml_pipeline_run_name",
            context.pipeline_run.name,
        )

        mlflow.log_param(
            "production_readiness_passed",
            production_readiness_passed,
        )

        mlflow.log_metric(
            "performance_gate_passed",
            int(performance_gate_passed),
        )

        mlflow.log_metric(
            "fairness_gate_passed",
            int(fairness_gate_passed),
        )

        mlflow.log_metric(
            "robustness_gate_passed",
            int(robustness_gate_passed),
        )

        mlflow.log_metric(
            "metamorphic_gate_passed",
            int(metamorphic_gate_passed),
        )

        mlflow.log_artifact(html_report_path)
        mlflow.log_artifact(json_report_path)

    print("\n" + "=" * 60)
    print("Production Readiness Quality Report")
    print("=" * 60)
    print(f"Report created: {html_report_path}")
    print(f"Overall Status: {'PASSED' if production_readiness_passed else 'FAILED'}")
    print("=" * 60)

    return quality_report, production_readiness_passed