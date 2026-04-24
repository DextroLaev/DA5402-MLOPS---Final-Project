import json
import os
import mlflow
import config

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

best_file = os.path.join(config.CHECKPOINT_DIR, "best_run.json")
with open(best_file, "r") as f:
    best = json.load(f)

run_id = best["run_id"]
print(f"Registering best model from sweep:")
print(f"  run_id    : {run_id}")
print(f"  val_loss  : {best['val_loss']:.4f}")
print(f"  params    : {best['params']}")

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

print(f"Model version {result.version} registered and promoted to Staging")

# ── Write version file (DVC tracks this as output of 'register' stage) ────────
version_file = os.path.join(config.CHECKPOINT_DIR, "registered_version.txt")
with open(version_file, "w") as f:
    f.write(f"{result.version}\n")
print(f"Wrote {version_file}")