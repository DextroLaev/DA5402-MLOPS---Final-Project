import json
import os
import mlflow
import config
import logging

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

DATASET_NAME = os.environ.get("DATASET_NAME", "lfw")
best_file = os.path.join(config.CHECKPOINT_DIR,f"{DATASET_NAME}_best_run.json")
with open(best_file, "r") as f:
    best = json.load(f)

run_id = best["run_id"]
log.info(f"Registering best model from sweep:")
log.info(f"  run_id    : {run_id}")
log.info(f"  val_loss  : {best['val_loss']:.4f}")
log.info(f"  params    : {best['params']}")

result = mlflow.register_model(
    model_uri=f"runs:/{run_id}/best_model",
    name="SiameseFaceRecognition",
)

client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="SiameseFaceRecognition",
    version=result.version,
    stage="Staging",
)

log.info(f"Model version {result.version} registered and promoted to Staging")

# ── Write version file (DVC tracks this as output of 'register' stage) ────────
version_file = os.path.join(config.CHECKPOINT_DIR, f"{DATASET_NAME}_registered_version.txt")
with open(version_file, "w") as f:
    f.write(f"{result.version}\n")
log.info(f"Wrote {version_file}")