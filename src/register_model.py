import json
import os
import mlflow
import config

mlflow.set_tracking_uri("http://mlflow:5000")  
# mlflow.set_tracking_uri(os.environ["REMOTE_MLFLOW_URI"])

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

print(f"Model version {result.version} promoted to Production")
