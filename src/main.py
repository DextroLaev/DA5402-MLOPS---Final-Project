import os
import config
import logging

log = logging.getLogger(__name__)

import torch
import logging
from dataloader import get_dataloaders
from model import build_model
from train import train
import json
import yaml
import mlflow
from mlflow.tracking import MlflowClient




def _load_best_params_from_json():
    """
    Read the best params written by sweep_optuna.py.
    Returns an empty dict if the file doesn't exist yet (first-ever run).
    """
    path = os.path.join(config.CHECKPOINT_DIR, f"{config.DATASET_NAME}_best_run.json")
    if not os.path.exists(path):
        log.warning(f"[main] No sweep JSON found at {path} — will use params.yaml / config defaults.")
        return {}
    with open(path) as f:
        data = json.load(f)
    params = data.get("params", {})
    sweep_id = data.get("sweep_id", "unknown")
    log.info(f"[main] Loaded best params from sweep '{sweep_id}': {params}")
    return params

def _load_params_from_yaml():
    """
    Fall-back: read the [train] block from params.yaml.
    Uses the midpoint of lr / margin ranges as the concrete value.
    """
    yaml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "params.yaml"
    )
    if not os.path.exists(yaml_path):
        return {}
    with open(yaml_path) as f:
        p = yaml.safe_load(f).get("train", {})
 
    mining_raw = p.get("mining_choices", config.TRIPLET_MINING)
    mining = mining_raw.split(",")[0].strip() if isinstance(mining_raw, str) else mining_raw[0]
 
    return {
        "learning_rate":  (p.get("lr_min", config.LEARNING_RATE) + p.get("lr_max", config.LEARNING_RATE)) / 2,
        "margin":         (p.get("margin_min", config.MARGIN) + p.get("margin_max", config.MARGIN)) / 2,
        "triplet_mining": mining,
        "warmup_epochs":  p.get("warmup_epochs", config.WARMUP_EPOCHS),
        "batch_size":     p.get("batch_size", config.BATCH_SIZE),
    }

def resolve_params() -> dict:
    """
        Merge param sources with priority:
        sweep JSON  >  params.yaml  >  config defaults
    """
    yaml_params  = _load_params_from_yaml()
    sweep_params = _load_best_params_from_json()
 
    merged = {
        "learning_rate":  config.LEARNING_RATE,
        "margin":         config.MARGIN,
        "triplet_mining": config.TRIPLET_MINING,
        "warmup_epochs":  config.WARMUP_EPOCHS,
    }
    merged.update(yaml_params)   
    merged.update(sweep_params)  
    return merged

def apply_params(params: dict) -> None:
    """Push resolved params back into the live config module."""
    config.LEARNING_RATE  = float(params["learning_rate"])
    config.MARGIN = float(params["margin"])
    config.TRIPLET_MINING = str(params["triplet_mining"])
    config.WARMUP_EPOCHS = int(params["warmup_epochs"])
    config.BATCH_SIZE = int(params["batch_size"])
    log.info(
        f"[main] Active params - lr={config.LEARNING_RATE:.2e}  "
        f"margin={config.MARGIN:.3f}  mining={config.TRIPLET_MINING}  "
        f"warmup={config.WARMUP_EPOCHS}"
        f"Batch Siz={config.BATCH_SIZE}"
    )


def load_production_weights(model, mlflow_uri: str, model_name: str):
    """
    Pull the current Production model from MLflow and warm-start *model* with
    its weights.  If no Production model exists yet, returns *model* unchanged
    (cold start).
    """

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()

    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not prod_versions:
        print("[main] No Production model registered yet — training from scratch.")
        return model

    uri = f"models:/{model_name}/Production"
    log.info(f"[main] Loading Production weights from {uri} (v{prod_versions[0].version})")
    prod_model = mlflow.pytorch.load_model(uri, map_location="cpu")
    missing, unexpected = model.load_state_dict(prod_model.state_dict(), strict=False)
    if missing or unexpected:
        log.warning(f"[main] missing keys  : {missing}")
        log.warning(f"[main] unexpected keys: {unexpected}")
    log.info("[main] Production weights loaded - finetuning mode active.")
    return model
 

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    params = resolve_params()
    apply_params(params)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config.BATCH_SIZE, k=4
    )
 
    model = build_model()
 
 
    mlflow_uri = os.environ.get("REMOTE_MLFLOW_URI", "").strip()
    model_name = os.environ.get("MLFLOW_MODEL_NAME", "SiameseFaceRecognition")
    triggered_by = os.environ.get("TRIGGERED_BY", "manual")
 
    if mlflow_uri:
        model = load_production_weights(model, mlflow_uri, model_name)
    else:
        reason = "no MLFLOW_TRACKING_URI" if not mlflow_uri else f"triggered_by={triggered_by!r}"
        log.info(f"[main] Skipping production-weight load ({reason}) - cold start.")
 
    resume_path = os.path.join(config.CHECKPOINT_DIR, config.LAST_CKPT_NAME)
    resume_path = resume_path if os.path.exists(resume_path) else None
 
 
    log.info("[main] Training started …")
    best_val_loss, run_id = train(
        model, train_loader, val_loader, device, resume_path=resume_path
    )
 

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
 
    best_run = {
        "sweep_id": "no-sweep",
        "run_id":   run_id,
        "val_loss": best_val_loss,
        "params":   params, 
    }
    best_run_path = os.path.join(
        config.CHECKPOINT_DIR, f"{config.DATASET_NAME}_best_run.json"
    )
    with open(best_run_path, "w") as f:
        json.dump(best_run, f, indent=2)
 
    mlflow_metrics = {}

    if mlflow_uri:
        try:
            mlflow_metrics = mlflow.get_run(run_id).data.metrics
        except Exception as exc:
            log.error(f"[main] Could not fetch MLflow metrics for run {run_id}: {exc}",exc_info=True)
 
    data_metrics_path = os.path.join(
        config.DATA_DIR, f"{config.DATASET_NAME}_data_metrics.json"
    )
    data_metrics = {}
    if os.path.exists(data_metrics_path):
        with open(data_metrics_path) as f:
            data_metrics = json.load(f)
 
    _params_prepare = {}

    _yaml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "params.yaml"
    )

    if os.path.exists(_yaml_path):
        with open(_yaml_path) as f:
            _params_prepare = yaml.safe_load(f).get("prepare", {})
 
    dvc_metrics = {
        "dataset_name":  config.DATASET_NAME,
        "n_trials":      1,
        "best_trial":    0,
        "val_loss":      mlflow_metrics.get("val_loss",   best_val_loss),
        "train_loss":    mlflow_metrics.get("train_loss"),
        "val_acc":       mlflow_metrics.get("val_acc"),
        "train_acc":     mlflow_metrics.get("train_acc"),
        "lr":            config.LEARNING_RATE,
        "margin":        config.MARGIN,
        "mining":        config.TRIPLET_MINING,
        "epochs":        config.NUM_EPOCHS,
        "batch_size":    config.BATCH_SIZE,
        "train_ratio":   _params_prepare.get("train_ratio"),
        "val_ratio":     _params_prepare.get("val_ratio"),
       
        **data_metrics,
    }
 
    dvc_path = os.path.join(
        config.CHECKPOINT_DIR, f"{config.DATASET_NAME}_dvc_metrics.json"
    )
    with open(dvc_path, "w") as f:
        json.dump(dvc_metrics, f, indent=2)
 
    log.info(f"[main] Done.  val_loss={best_val_loss:.4f}  run_id={run_id}")
 