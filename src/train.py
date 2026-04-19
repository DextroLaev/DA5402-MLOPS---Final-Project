import json
import os
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from utils import TripletLoss

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

        if n_active < config.MIN_ACTIVE_TRIPLETS:
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        tot_loss += loss.item()
        tot_acc  += batch_accuracy(embeddings.detach(), labels)
        steps    += 1

    if steps == 0:
        return 0.0, 0.0, 0.0
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
        return 0.0, 0.0, 0.0
    return tot_loss / steps, tot_acc / steps


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

def train(model, train_loader, val_loader, device, resume_path=None):
    criterion = TripletLoss(margin=config.MARGIN, mining=config.TRIPLET_MINING)
    phase = 1
    optimizer = make_optimizer(model, phase)
    scheduler = CosineAnnealingLR(optimizer, T_max=WARMUP_EPOCHS, eta_min=config.LEARNING_RATE * 0.1)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    start_epoch = 1
 
    if resume_path and os.path.exists(resume_path):
        start_epoch, phase, best_val_loss, history = load_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch - 1}, phase {phase}")
 
    model.to(device)
 
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
 
        if epoch == WARMUP_EPOCHS + 1 and phase == 1:
            print("Switching to phase 2 - unfreezing backbone")
            for p in model.encoder.backbone.parameters():
                p.requires_grad = True
            phase = 2
            optimizer = make_optimizer(model, phase)
            scheduler = CosineAnnealingLR(optimizer,T_max  = config.NUM_EPOCHS - WARMUP_EPOCHS,eta_min= FINETUNE_LR * 0.01,)
 
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
 
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
 
        print(
            f"[phase {phase}] Epoch {epoch:03d}/{config.NUM_EPOCHS} | "
            f"train loss={tr_loss:.4f} acc={tr_acc*100:.1f}% | "
            f"val loss={vl_loss:.4f} acc={vl_acc*100:.1f}% | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s"
        )
 
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            save_checkpoint(model, optimizer, scheduler, epoch, phase, best_val_loss, history, tag="best")
            print(f"Best model saved with val loss={vl_loss:.4f}")
 
        if epoch % config.SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, phase, best_val_loss, history, tag="last")
 
    save_checkpoint(model, optimizer, scheduler, config.NUM_EPOCHS, phase, best_val_loss, history, tag="last")
 
    with open(os.path.join(config.LOG_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
 
    print(f"Training done. Best val loss: {best_val_loss:.4f}")