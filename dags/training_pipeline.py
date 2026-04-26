import os
import sqlite3
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.models import Variable
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.bash import BashOperator
from docker.types import Mount, DeviceRequest
import mlflow

log = logging.getLogger(__name__)

HOST_PROJECT_DIR = os.environ["HOST_PROJECT_DIR"]
RUN_REMOTE = os.environ.get('RUN_REMOTE', "false") == 'true'

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

def run_prepare_data(**context):
    """
    DAG wrapper for prepare_data.py.
    Validates the dataset, merges misclassified crops, computes baseline stats.
    """
    import subprocess
    result = subprocess.run(
        ["python", "/opt/airflow/src/prepare_data.py"],
        capture_output=True, text=True
    )
    log.info(result.stdout)
    if result.returncode != 0:
        log.error(result.stderr)
        raise RuntimeError(f"prepare_data.py failed:\n{result.stderr}")
    log.info("Data preparation complete.")


def export_misclassified_data(**context):
    """
    When triggered by Flask's misclassification threshold,
    dump corrected face crops from SQLite onto disk so the trainer picks them up.
    For scheduled / manual runs this is a no-op.
    """
    conf = context["dag_run"].conf or {}
    triggered_by = conf.get("triggered_by", "schedule")

    if triggered_by != "misclassification_threshold":
        log.info(f"Run triggered by {triggered_by!r}; skipping misclassified export.")
        context["ti"].xcom_push(key="export_total", value=0)
        context["ti"].xcom_push(key="triggered_by", value=triggered_by)
        return {"exported": 0}

    db_path  = _cfg("DB_PATH", "/opt/airflow/data/face_db.sqlite")
    out_root = _cfg("MISCLASSIFIED_DIR", "/opt/airflow/data/misclassified")

    con  = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT true_name, face_bytes, timestamp FROM misclassified_faces ORDER BY timestamp"
    ).fetchall()
    con.close()

    if not rows:
        log.info("Misclassification-triggered run, but no face crops found.")
        context["ti"].xcom_push(key="export_total", value=0)
        context["ti"].xcom_push(key="triggered_by", value=triggered_by)
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
    context["ti"].xcom_push(key="triggered_by", value=triggered_by)
    if total > 0:
        con = sqlite3.connect(db_path)
        con.execute("DELETE FROM misclassified_faces")
        con.commit()
        con.close()
        log.info(f"Cleared misclassified_faces table after exporting {total} crops.")
    return {"exported": total}


def evaluate_and_promote(**context):
    """
    Compare the latest Staging model with the current Production model.
    Promote Staging → Production if it wins, or if there is no Production yet.
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


def check_should_train(**context) -> bool:
    """
    ShortCircuitOperator callable.
    Returns True  → pipeline continues (train + register + promote).
    Returns False → all downstream tasks are SKIPPED.

    Rules:
      - triggered by misclassification_threshold → ALWAYS train
      - triggered by schedule with no misclassifications → SKIP
      - any other trigger (manual) → ALWAYS train
    """
    conf         = context["dag_run"].conf or {}
    triggered_by = conf.get("triggered_by", "schedule")

    if triggered_by == "misclassification_threshold":
        log.info("Triggered by misclassification — proceeding with training.")
        return True

    if triggered_by != "schedule":
        log.info(f"Triggered by {triggered_by!r} — proceeding with training.")
        return True

    db_path = _cfg("DB_PATH", "/opt/airflow/data/face_db.sqlite")
    if not os.path.exists(db_path):
        log.info("Scheduled run: no DB found — skipping training.")
        return False

    con   = sqlite3.connect(db_path)
    count = con.execute("SELECT COUNT(*) FROM misclassified_faces").fetchone()[0]
    con.close()

    if count > 0:
        log.info(f"Scheduled run: {count} pending misclassifications found — proceeding.")
        return True

    log.info("Scheduled run: no misclassifications — skipping training.")
    return False


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


# shared train variables

_COMMON_TRAIN_ENV = {
    "MLFLOW_TRACKING_URI": _cfg("REMOTE_MLFLOW_URI", "http://mlflow:5000"),
    "MLFLOW_MODEL_NAME":   _cfg("MLFLOW_MODEL_NAME",   "SiameseFaceRecognition"),
    "TRIGGERED_BY":        "{{ dag_run.conf.get('triggered_by', 'schedule') }}",
    "DATASET_NAME":        "{{ dag_run.conf.get('dataset', params.dataset) }}",
    "NUM_EPOCHS":          "25",
    "PYTHONUNBUFFERED":    "1",
    "GIT_PYTHON_REFRESH":  "quiet",
}

# ── Unified DAG ──────────────────────────────────────────────────────────────

with DAG(
    dag_id="face_recognition_pipeline",
    description=(
        "Initial training + misclassification-triggered retraining. "
        "Runs weekly by default; Flask also triggers it when the unique "
        "misclassification count crosses its threshold. "
        "Uses main.py with params.yaml hyperparameters — single train, no sweep."
    ),
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["face-recognition", "mlops"],
    params={    
        "dataset":        Param("lfw",   type="string"),
        "learning_rate":  Param(1e-5,    type="number"),  
        "margin":         Param(0.2,     type="number"), 
        "triplet_mining": Param("semi",  type="string"),
        "warmup_epochs":  Param(5,       type="integer"),
        "batch_size":     Param(32,      type="integer"),
    },
) as dag:

    prepare_data = PythonOperator(
        task_id="prepare_data",
        python_callable=run_prepare_data,
    )

    export_misclassified = PythonOperator(
        task_id="export_misclassified_data",
        python_callable=export_misclassified_data,
    )

    should_train = ShortCircuitOperator(
        task_id="check_should_train",
        python_callable=check_should_train,
        ignore_downstream_trigger_rules=True,
    )

    if RUN_REMOTE:
        train_task = BashOperator(
            task_id="train_model",
            bash_command="""
            set -euo pipefail

            KEY_SRC=/opt/airflow/ssh_keys/id_rsa
            if [ ! -f "$KEY_SRC" ]; then
                echo "ERROR: SSH key not found at $KEY_SRC" >&2
                exit 1
            fi

            mkdir -p ~/.ssh && chmod 700 ~/.ssh
            cp "$KEY_SRC" ~/.ssh/id_rsa
            chmod 600 ~/.ssh/id_rsa
            trap 'rm -f ~/.ssh/id_rsa' EXIT

            SSH_OPTS="-i ~/.ssh/id_rsa -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PasswordAuthentication=no"

            LOCAL_MISCLASSIFIED="/opt/airflow/data/misclassified"
            REMOTE_MISCLASSIFIED="${REMOTE_DATA_DIR}/misclassified"

            # if [ "${TRIGGERED_BY}" = "misclassification_threshold" ] && [ -d "$LOCAL_MISCLASSIFIED" ] && [ "$(ls -A $LOCAL_MISCLASSIFIED 2>/dev/null)" ]; then
            if [ -d "$LOCAL_MISCLASSIFIED" ] && [ "$(ls -A $LOCAL_MISCLASSIFIED 2>/dev/null)" ]; then
                echo "Syncing misclassified crops to remote server..."
                rsync -avz --mkpath \
                    -e "ssh $SSH_OPTS" \
                    "$LOCAL_MISCLASSIFIED/" \
                    "${REMOTE_SSH_USER}@${REMOTE_SSH_HOST}:${REMOTE_MISCLASSIFIED}/"
                echo "Sync complete."
            else
                echo "No misclassified data to sync (triggered_by=${TRIGGERED_BY}) — skipping rsync."
            fi

            CONTAINER_NAME="trainer_$(echo ${AIRFLOW_CTX_DAG_RUN_ID:-$(date +%s)} | tr -cd 'a-zA-Z0-9_.-')"

            ssh -tt $SSH_OPTS \
                "${REMOTE_SSH_USER}@${REMOTE_SSH_HOST}" \
                "trap 'docker rm -f ${CONTAINER_NAME} 2>/dev/null || true' EXIT INT TERM HUP; \
                docker run --rm --gpus all --shm-size=4g --name ${CONTAINER_NAME} \
                -e MLFLOW_TRACKING_URI=${REMOTE_MLFLOW_URI} \
                -e REMOTE_MLFLOW_URI=${REMOTE_MLFLOW_URI} \
                -e MLFLOW_MODEL_NAME=${MLFLOW_MODEL_NAME} \
                -e TRIGGERED_BY=${TRIGGERED_BY} \
                -e DATASET_NAME=${DATASET_NAME} \
                -e NUM_EPOCHS=25 \
                -e PYTHONUNBUFFERED=1 \
                -e GIT_PYTHON_REFRESH=quiet \
                -v ${REMOTE_CODE_DIR}:/app \
                -v ${REMOTE_DATA_DIR}:/app/data \
                dextrolaev/trainer:latest python src/main.py"
            """,
            env={
                "REMOTE_SSH_USER":   os.environ.get("REMOTE_SSH_USER", "user"),
                "REMOTE_SSH_HOST":   os.environ.get("REMOTE_SSH_HOST", ""),
                "REMOTE_MLFLOW_URI": os.environ.get("REMOTE_MLFLOW_URI", ""),
                "REMOTE_CODE_DIR":   os.environ.get("REMOTE_CODE_DIR", "/home/user/code"),
                "REMOTE_DATA_DIR":   os.environ.get("REMOTE_DATA_DIR", "/home/user/data"),
                "MLFLOW_MODEL_NAME": os.environ.get("MLFLOW_MODEL_NAME", "SiameseFaceRecognition"),
                "TRIGGERED_BY":      "{{ dag_run.conf.get('triggered_by', 'schedule') }}",
                "DATASET_NAME":      "{{ dag_run.conf.get('dataset', params.dataset) }}",
            },
            append_env=True,
        )
    else:
        train_task = DockerOperator(
            task_id="train_model",
            image="trainer:latest",
            command="python src/main.py",
            network_mode="finalproject_mlops-net",
            auto_remove="force",
            docker_url="unix://var/run/docker.sock",
            shm_size=2 * 1024 * 1024 * 1024,
            device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
            mounts=[
                Mount(source=f"{HOST_PROJECT_DIR}/data",        target="/app/data",        type="bind"),
                Mount(source=f"{HOST_PROJECT_DIR}/checkpoints", target="/app/checkpoints", type="bind"),
                Mount(source=f"{HOST_PROJECT_DIR}/logs",        target="/app/logs",        type="bind"),
                Mount(source=f"{HOST_PROJECT_DIR}/params.yaml", target="/app/params.yaml", type="bind"),
            ],
            environment=_COMMON_TRAIN_ENV,
            execution_timeout=timedelta(hours=3),
        )

    register_model = BashOperator(
        task_id="register_model",
        bash_command="python /opt/airflow/src/register_model.py",
        env={
            "MLFLOW_TRACKING_URI": _cfg("REMOTE_MLFLOW_URI", "http://mlflow:5000"),
            "DATASET_NAME":        "{{ dag_run.conf.get('dataset', params.dataset) }}",
        },
        append_env=True,
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

    prepare_data >> export_misclassified >> should_train >> train_task
    train_task >> register_model >> promote_model >> notify >> health_check
    should_train >> health_check