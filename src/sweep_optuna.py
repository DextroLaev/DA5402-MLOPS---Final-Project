"""Optuna-driven hyperparameter sweep for face-recognition training.

Each trial becomes its own top-level MLflow run, tagged with a shared
`sweep_id`. In the MLflow UI: Experiments → face-recognition-siamese, then
filter / sort by `tags.sweep_id = "..."` (or just sort by val_loss) to
compare a sweep's trials.

At the end, the best trial's run id is written to checkpoints/best_run.json
so the downstream register_model task can promote only the winner.

Env vars (injected by the Airflow DAG):
  N_TRIALS                number of Optuna trials (default 5)
  N_EPOCHS_PER_TRIAL      epochs each trial trains for (default 3)
  LR_MIN / LR_MAX         learning-rate search range, log-uniform
  MARGIN_MIN / MARGIN_MAX triplet margin search range
  MINING_CHOICES          comma-separated categorical, e.g. "semi,hard"
"""
import json
import os
import uuid
from datetime import datetime, timezone

import optuna
import torch
import mlflow
from mlflow.tracking import MlflowClient

import config
from dataloader import get_dataloaders
from model import build_model
from train import train
import yaml
import numpy as np
import random

_params_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "params.yaml")
with open(_params_path) as f:
    _p = yaml.safe_load(f)["train"]

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)

N_TRIALS = int(os.environ.get("N_TRIALS",str(_p["n_trials"])))
N_EPOCHS_PER_TRIAL = int(os.environ.get("N_EPOCHS_PER_TRIAL", str(_p["n_epochs_per_trial"])))
LR_MIN = float(os.environ.get("LR_MIN",str(_p["lr_min"])))
LR_MAX = float(os.environ.get("LR_MAX",str(_p["lr_max"])))
MARGIN_MIN = float(os.environ.get("MARGIN_MIN",str(_p["margin_min"])))
MARGIN_MAX = float(os.environ.get("MARGIN_MAX",str(_p["margin_max"])))
MINING_CHOICES = os.environ.get("MINING_CHOICES",_p["mining_choices"]).split(",")
TRIGGERED_BY = os.environ.get("TRIGGERED_BY", "schedule")
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "SiameseFaceRecognition")

if TRIGGERED_BY == "misclassification_threshold":
    N_TRIALS = 1

def _load_production_state_dict(client: MlflowClient):
    """Fetch the current Production model's state_dict, or None if absent."""
    try:
        prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    except Exception as exc:
        print(f"[finetune] Could not query Production model: {exc}")
        return None
    if not prod:
        print("[finetune] No Production model registered yet — training from scratch.")
        return None
    uri = f"models:/{MODEL_NAME}/Production"
    print(f"[finetune] Loading Production weights from {uri} (v{prod[0].version})")
    loaded = mlflow.pytorch.load_model(uri, map_location="cpu")
    return loaded.state_dict()

SWEEP_ID = os.environ.get(
    "SWEEP_ID",
    f"sweep-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}",
)


def main():
    # mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("face-recognition-siamese")
    client = MlflowClient()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Sweep ID    : {SWEEP_ID}")
    print(f"Sweep config: {N_TRIALS} trials x {N_EPOCHS_PER_TRIAL} epochs each")

    train_loader, val_loader, _ = get_dataloaders(batch_size=config.BATCH_SIZE, k=4)

    finetune_state = None
    if TRIGGERED_BY == "misclassification_threshold":
        finetune_state = _load_production_state_dict(client)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", LR_MIN, LR_MAX, log=True)
        margin = trial.suggest_float("margin", MARGIN_MIN, MARGIN_MAX)
        mining = trial.suggest_categorical("triplet_mining", MINING_CHOICES)
        warmup = trial.suggest_int(
            "warmup_epochs", 1, max(1, N_EPOCHS_PER_TRIAL - 1)
        )

        config.LEARNING_RATE = lr
        config.MARGIN = margin
        config.TRIPLET_MINING = mining
        config.WARMUP_EPOCHS = warmup
        config.NUM_EPOCHS = N_EPOCHS_PER_TRIAL

        model = build_model()
        if finetune_state is not None:
            missing, unexpected = model.load_state_dict(finetune_state, strict=False)
            if missing or unexpected:
                print(f"[finetune] load_state_dict missing={missing} unexpected={unexpected}")
            for p in model.encoder.backbone.parameters():
                p.requires_grad = True
            print("[finetune] Initialized trial from Production weights (backbone unfrozen).")

        print(
            f"\n--- Trial {trial.number}: lr={lr:.2e} margin={margin:.3f} "
            f"mining={mining} warmup={warmup} ---"
        )
        try:
            best_val_loss, run_id = train(
                model, train_loader, val_loader, device,
                resume_path=None, mlflow_nested=False,
            )
        finally:
            del model
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        client.set_tag(run_id, "sweep_id", SWEEP_ID)
        client.set_tag(run_id, "trial_index", str(trial.number))
        client.set_tag(run_id, "mlflow.runName", f"{SWEEP_ID}-trial-{trial.number}")

        trial.set_user_attr("mlflow_run_id", run_id)
        
        return best_val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

    best = study.best_trial
    best_run_id = best.user_attrs["mlflow_run_id"]

    client.set_tag(best_run_id, "best_of_sweep", "true")

    print("\n=== Sweep complete ===")
    print(f"Sweep ID    : {SWEEP_ID}")
    print(f"Best trial  : #{best.number}")
    print(f"  val_loss  : {best.value:.4f}")
    print(f"  params    : {best.params}")
    print(f"  run_id    : {best_run_id}")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    out = {
        "sweep_id": SWEEP_ID,
        "run_id": best_run_id,
        "val_loss": best.value,
        "params": best.params,
    }
    with open(os.path.join(config.CHECKPOINT_DIR, f"{config.DATASET_NAME}_best_run.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {config.CHECKPOINT_DIR}/best_run.json")

    best_mlflow_metrics = client.get_run(best_run_id).data.metrics

    data_metrics_path = os.path.join(config.DATA_DIR, f"{config.DATASET_NAME}_data_metrics.json")
    data_metrics = {}

    if os.path.exists(data_metrics_path):
        with open(data_metrics_path) as f:
            data_metrics = json.load(f)
    else:
        print(f"[sweep] Warning: data metrics not found at {data_metrics_path} — skipping merge.")
    
    _params_train = _p
    _params_prepare = yaml.safe_load(
        open(_params_path).read()
    ).get("prepare", {})

    dvc_metrics = {
        "dataset_name":  config.DATASET_NAME,
        "n_trials":      N_TRIALS,
        "best_trial":    best.number,
        "val_loss":      best_mlflow_metrics.get("val_loss",   best.value),
        "train_loss":    best_mlflow_metrics.get("train_loss"),
        "val_acc":       best_mlflow_metrics.get("val_acc"),
        "train_acc":     best_mlflow_metrics.get("train_acc"),
        "lr":            best.params.get("learning_rate"),
        "margin":        best.params.get("margin"),
        "mining":        best.params.get("triplet_mining"),
        "epochs":        N_EPOCHS_PER_TRIAL,
        "batch_size":    config.BATCH_SIZE,
        "train_ratio":   _params_prepare.get("train_ratio"),
        "val_ratio":     _params_prepare.get("val_ratio"),

        **data_metrics,
    }

    dvc_path = os.path.join(config.CHECKPOINT_DIR, f"{config.DATASET_NAME}_dvc_metrics.json")
    
    with open(dvc_path, "w") as f:
        json.dump(dvc_metrics, f, indent=2)
    print(f"[sweep] Wrote consolidated metrics → {dvc_path}")
 
 
if __name__ == "__main__":
    main()