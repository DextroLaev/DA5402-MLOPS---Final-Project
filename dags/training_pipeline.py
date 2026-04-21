import os
import sqlite3
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.bash import BashOperator
from docker.types import Mount, DeviceRequest
import mlflow

log = logging.getLogger(__name__)

HOST_PROJECT_DIR = os.environ["HOST_PROJECT_DIR"]

default_args = {
    "owner":       "mlops",
    "retries":     1,
    "retry_delay": timedelta(minutes=5),
}


def _cfg(key: str, default: str = "") -> str:
    try:
        return Variable.get(key)
    except Exception:
        return os.environ.get(key, default)


# ── Task callables ───────────────────────────────────────────────────────────

def export_misclassified_data(**context):
    """
    When this DAG run was triggered by Flask's misclassification threshold,
    dump corrected face crops from SQLite onto disk so the trainer picks them up.
    For scheduled / initial runs there is nothing to export — this is a no-op.
    """
    conf         = context["dag_run"].conf or {}
    triggered_by = conf.get("triggered_by", "schedule")

    if triggered_by != "misclassification_threshold":
        log.info(f"Run triggered by {triggered_by!r}; skipping misclassified export.")
        context["ti"].xcom_push(key="export_total", value=0)
        return {"exported": 0}

    db_path  = _cfg("DB_PATH", "/opt/airflow/data/face_db.sqlite")
    out_root = os.path.join(HOST_PROJECT_DIR, "data", "misclassified")

    con  = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT true_name, face_bytes, timestamp FROM misclassified_faces ORDER BY timestamp"
    ).fetchall()
    con.close()

    if not rows:
        log.info("Misclassification-triggered run, but no face crops found.")
        context["ti"].xcom_push(key="export_total", value=0)
        return {"exported": 0}

    counts: dict[str, int] = {}
    for true_name, face_bytes, ts in rows:
        person_dir = os.path.join(out_root, true_name)
        os.makedirs(person_dir, exist_ok=True)
        idx = counts.get(true_name, 0)
        with open(os.path.join(person_dir, f"{int(ts)}_{idx}.jpg"), "wb") as f:
            f.write(face_bytes)
        counts[true_name] = idx + 1

    total = sum(counts.values())
    log.info(f"Exported {total} crops for {len(counts)} persons → {out_root}")
    context["ti"].xcom_push(key="export_total", value=total)
    return {"exported": total}


def evaluate_and_promote(**context):
    """
    Compare the latest Staging model with the current Production model.
    Promote Staging → Production if it wins, or if there is no Production yet
    (the initial training run).
    """
    mlflow_uri    = _cfg("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_name    = _cfg("MLFLOW_MODEL_NAME",   "SiameseFaceRecognition")
    eval_metric   = _cfg("EVAL_METRIC",         "val_loss")
    higher_better = _cfg("HIGHER_IS_BETTER",    "false").lower() == "true"

    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()

    staging = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging:
        log.warning("No Staging model found — nothing to promote.")
        context["ti"].xcom_push(key="promoted", value=False)
        return {"promoted": False}

    new_mv     = staging[0]
    new_ver    = new_mv.version
    new_run    = client.get_run(new_mv.run_id)
    new_metric = new_run.data.metrics.get(eval_metric)
    log.info(f"Staging model v{new_ver} {eval_metric}={new_metric}")

    prod    = client.get_latest_versions(model_name, stages=["Production"])
    promote = True

    if prod and new_metric is not None:
        prod_run    = client.get_run(prod[0].run_id)
        prod_metric = prod_run.data.metrics.get(eval_metric)
        log.info(f"Production model v{prod[0].version} {eval_metric}={prod_metric}")
        if prod_metric is not None:
            promote = (new_metric > prod_metric) if higher_better else (new_metric < prod_metric)

    log.info(f"Promote? {promote}")

    if promote:
        for pv in prod:
            log.info(f"Archiving old Production v{pv.version}")
            client.transition_model_version_stage(
                name=model_name, version=pv.version, stage="Archived"
            )
        client.transition_model_version_stage(
            name=model_name, version=new_ver, stage="Production"
        )
        log.info(f"Model v{new_ver} promoted to Production")

    context["ti"].xcom_push(key="promoted",    value=promote)
    context["ti"].xcom_push(key="new_version", value=new_ver)
    return {"promoted": promote, "version": new_ver}


def notify_flask(**context):
    """
    Ping Flask /ready so the hot-reload thread picks up the new Production
    model immediately instead of waiting for its next poll.
    """
    import requests

    flask_url = _cfg("FLASK_API_URL", "http://flask-api:2000")
    promoted  = context["ti"].xcom_pull(task_ids="evaluate_and_promote", key="promoted")
    version   = context["ti"].xcom_pull(task_ids="evaluate_and_promote", key="new_version")

    if not promoted:
        log.info("No promotion — Flask will pick up the model on its next poll.")
        return

    try:
        resp = requests.get(f"{flask_url}/ready", timeout=5)
        log.info(f"Flask /ready → {resp.status_code} model_version={version}")
    except Exception as exc:
        log.warning(f"Could not reach Flask /ready: {exc}")


# ── Unified DAG ──────────────────────────────────────────────────────────────

with DAG(
    dag_id="face_recognition_pipeline",
    description=(
        "Initial training + misclassification-triggered retraining. "
        "Runs weekly by default; Flask also triggers it when the unique "
        "misclassification count crosses its threshold."
    ),
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["face-recognition", "mlops"],
    params={
        "n_trials":           Param(5,    type="integer"),
        "n_epochs_per_trial": Param(3,    type="integer"),
        "lr_min":             Param(1e-5, type="number"),
        "lr_max":             Param(1e-2, type="number"),
        "margin_min":         Param(0.2,  type="number"),
        "margin_max":         Param(1.0,  type="number"),
        "mining_choices":     Param("semi,hard", type="string"),
    },
) as dag:

    export_misclassified = PythonOperator(
        task_id="export_misclassified_data",
        python_callable=export_misclassified_data,
    )

    train_model = DockerOperator(
        task_id="train_model",
        image="trainer:latest",
        command="python src/sweep_optuna.py",
        network_mode="finalproject_mlops-net",
        auto_remove="success",
        docker_url="unix://var/run/docker.sock",
        mount_tmp_dir=False,
        shm_size=2 * 1024 * 1024 * 1024,
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
        mounts=[
            Mount(source=f"{HOST_PROJECT_DIR}/data",        target="/app/data",        type="bind"),
            Mount(source=f"{HOST_PROJECT_DIR}/checkpoints", target="/app/checkpoints", type="bind"),
            Mount(source=f"{HOST_PROJECT_DIR}/logs",        target="/app/logs",        type="bind"),
        ],
        environment={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "TRIGGERED_BY":        "{{ dag_run.conf.get('triggered_by', 'schedule') }}",
            "N_TRIALS":            "{{ params.n_trials }}",
            "N_EPOCHS_PER_TRIAL":  "{{ params.n_epochs_per_trial }}",
            "LR_MIN":              "{{ params.lr_min }}",
            "LR_MAX":              "{{ params.lr_max }}",
            "MARGIN_MIN":          "{{ params.margin_min }}",
            "MARGIN_MAX":          "{{ params.margin_max }}",
            "MINING_CHOICES":      "{{ params.mining_choices }}",
        },
        execution_timeout=timedelta(hours=3),
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command="python /opt/airflow/src/register_model.py",
    )

    promote_model = PythonOperator(
        task_id="evaluate_and_promote",
        python_callable=evaluate_and_promote,
    )

    notify = PythonOperator(
        task_id="notify_flask",
        python_callable=notify_flask,
    )

    health_check = BashOperator(
        task_id="api_health_check",
        bash_command="sleep 5 && curl -sf http://flask-api:2000/health && echo 'API healthy'",
    )

    export_misclassified >> train_model >> register_model >> promote_model >> notify >> health_check
