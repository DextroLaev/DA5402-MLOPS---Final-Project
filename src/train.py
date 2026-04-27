import json
import os
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import config
from utils import TripletLoss
import mlflow 
import mlflow.pytorch
import logging

log = logging.getLogger(__name__)


WARMUP_EPOCHS = config.WARMUP_EPOCHS
FINETUNE_LR   = config.FINETUNE_LR
BACKBONE_LR   = config.BACKBONE_LR

def batch_accuracy(embeddings, labels):
    dist = torch.cdist(embeddings, embeddings, p=2)
    eye  = torch.eye(dist.size(0), dtype=torch.bool, device=dist.device)
    dist = dist.masked_fill(eye, float("inf"))
    nn_labels = labels[dist.argmin(dim=1)]
    return (nn_labels == labels).float().mean().item()

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    tot_loss, tot_acc, tot_act, steps = 0.0, 0.0, 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        embeddings = model.encoder(images)
        loss, n_active = criterion(embeddings, labels)

        # if n_active < config.MIN_ACTIVE_TRIPLETS:
        #     del loss, embeddings
        #     continue

        if n_active == 0:
            # no valid triplets → give zero loss but still backprop safely
            loss = embeddings.sum() * 0.0

        try:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        except RuntimeError as exc:
            log.error("Backward pass failed at step %d: %s", steps, exc)
            optimizer.zero_grad()
            continue

        tot_loss += loss.item()
        tot_acc  += batch_accuracy(embeddings.detach(), labels)
        steps    += 1

    if steps == 0:
        return 0.0, 0.0
    return tot_loss / steps, tot_acc / steps


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    tot_loss, tot_acc, tot_act, steps = 0.0, 0.0, 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        embeddings = model.encoder(images)
        loss, n_active = criterion(embeddings, labels)

        tot_loss += loss.item()
        tot_acc  += batch_accuracy(embeddings, labels)
        steps    += 1

    if steps == 0:
        return 0.0, 0.0
    return tot_loss / steps, tot_acc / steps


@torch.no_grad()
def eval_mode_accuracy(model, loader, device):
    """
        Clean, eval-mode accuracy over a loader (no dropout, BN in eval stats).
    """
    was_training = model.training
    model.eval()
    tot_acc, steps = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        embeddings = model.encoder(images)
        tot_acc += batch_accuracy(embeddings, labels)
        steps += 1
    if was_training:
        model.train()
    return tot_acc / steps if steps else 0.0


def make_optimizer(model, phase):
    if phase == 1:
        params = [p for p in model.parameters() if p.requires_grad]
        return optim.Adam(params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    else:
        return optim.Adam([
            {"params": model.encoder.backbone.parameters(), "lr": BACKBONE_LR},
            {"params": model.encoder.embed.parameters(),     "lr": FINETUNE_LR},
        ], weight_decay=config.WEIGHT_DECAY)

def save_checkpoint(model, optimizer, scheduler, epoch, phase, best_val_loss, history, tag="last"):
    name = config.BEST_CKPT_NAME if tag == "best" else config.LAST_CKPT_NAME
    torch.save({
        "epoch":            epoch,
        "phase":            phase,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "sched_state_dict": scheduler.state_dict(),
        "best_val_loss":    best_val_loss,
        "history":          history,
    }, os.path.join(config.CHECKPOINT_DIR, name))

def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optim_state_dict"])
    scheduler.load_state_dict(ckpt["sched_state_dict"])
    return (
        ckpt["epoch"],
        ckpt.get("phase", 1),
        ckpt.get("best_val_loss", float("inf")),
        ckpt.get("history", {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}),
    )

def train(model, train_loader, val_loader, device, resume_path=None, mlflow_nested=False):

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment('face-recognition-siamese')

    with mlflow.start_run(nested=mlflow_nested) as run:
        mlflow.log_params({
            "batch_size": config.BATCH_SIZE,
            "num_epochs": config.NUM_EPOCHS,
            "learning_rate": config.LEARNING_RATE,
            'weight_decay':config.WEIGHT_DECAY,
            "margin": config.MARGIN,
            "embedding_dim": config.EMBEDDING_DIM,
            "triplet_mining": config.TRIPLET_MINING,
            "warmup_epochs": config.WARMUP_EPOCHS,
            'threshold':config.THRESHOLD
        })


        criterion = TripletLoss(margin=config.MARGIN, mining=config.TRIPLET_MINING)
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        start_epoch = 1
        phase = 1

        if resume_path and os.path.exists(resume_path):
            ckpt_phase = torch.load(resume_path, map_location="cpu").get("phase", 1)
            if ckpt_phase == 2:
                for p in model.encoder.backbone.parameters():
                    p.requires_grad = True
            phase = ckpt_phase

        optimizer = make_optimizer(model, phase)
        if phase == 1:
            scheduler = CosineAnnealingLR(optimizer, T_max=config.WARMUP_EPOCHS, eta_min=config.LEARNING_RATE * 0.1)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS - config.WARMUP_EPOCHS, eta_min=config.FINETUNE_LR * 0.01)

        if resume_path and os.path.exists(resume_path):
            start_epoch, phase, best_val_loss, history = load_checkpoint(
                resume_path, model, optimizer, scheduler, device
            )
            start_epoch += 1
            log.info(f"Resumed from epoch {start_epoch - 1}, phase {phase}")
    
        model.to(device)
    
        for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
    
            if epoch == config.WARMUP_EPOCHS + 1 and phase == 1:
                log.info("Switching to phase 2 - unfreezing backbone")
                for p in model.encoder.backbone.parameters():
                    p.requires_grad = True
                phase = 2
                optimizer = make_optimizer(model, phase)
                scheduler = CosineAnnealingLR(optimizer,T_max  = config.NUM_EPOCHS - config.WARMUP_EPOCHS,eta_min= config.FINETUNE_LR * 0.01,)
    
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            vl_loss, vl_acc = val_epoch(model, val_loader, criterion, device)
            scheduler.step()
            torch.cuda.empty_cache()
    
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(vl_loss)
            history["val_acc"].append(vl_acc)
    
            log.info(
                f"[phase {phase}] Epoch {epoch:03d}/{config.NUM_EPOCHS} | "
                f"train loss={tr_loss:.4f} train_acc={tr_acc*100:.1f}% | "
                f"val loss={vl_loss:.4f} val_acc={vl_acc*100:.1f}% | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s"
            )

            mlflow.log_metrics({
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "val_loss": vl_loss,
                    "val_acc": vl_acc,
                    "learning_rate": scheduler.get_last_lr()[0],
                }, step=epoch)
    
            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                save_checkpoint(model, optimizer, scheduler, epoch, phase, best_val_loss, history, tag="best")
                log.info(f"Best model saved with val loss={vl_loss:.4f}")

        best_ckpt_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_CKPT_NAME)
        if os.path.exists(best_ckpt_path):
            best_state = torch.load(best_ckpt_path, map_location=device)["model_state_dict"]
            model.load_state_dict(best_state)
            mlflow.pytorch.log_model(model, "best_model")
        
        model.cpu()
        torch.cuda.empty_cache()

        with open(os.path.join(config.LOG_DIR, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        try:
            mlflow.log_artifact(os.path.join(config.LOG_DIR, "history.json"))
        except Exception:
            log.warning("Could not log history artifact to MLflow", exc_info=True)

        # Save run ID so register_model.py can find it
        with open(os.path.join(config.LOG_DIR, "mlflow_run_id.txt"), "w") as f:
            f.write(run.info.run_id)

        log.info(f"Training done. Best val loss: {best_val_loss:.4f}")
        log.info(f"MLflow run ID: {run.info.run_id}")

        return best_val_loss, run.info.run_id