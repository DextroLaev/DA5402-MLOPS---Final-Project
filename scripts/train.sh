#!/bin/bash
set -euo pipefail

DATASET_NAME=${1:-lfw}

# ── Load .env ─────────────────────────────────────────────────────────────────
set -a
source "$(dirname "$0")/../.env"
set +a

# ─────────────────────────────────────────────────────────────────────────────
if [ "$RUN_REMOTE" = "true" ]; then
# ─────────────────────────────────────────────────────────────────────────────

    echo "[train.sh] MODE: REMOTE — training on ${REMOTE_SSH_HOST}"

    SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes"
    CONTAINER_NAME="dvc_trainer_${DATASET_NAME}_$(date +%s)"

    # 1. Sync code
    echo "[train.sh] Syncing code..."
    rsync -avz --exclude='.git' --exclude='data/' --exclude='checkpoints/' \
        -e "ssh $SSH_OPTS" \
        ./ "${REMOTE_SSH_USER}@${REMOTE_SSH_HOST}:${REMOTE_CODE_DIR}/"

    # 2. Sync misclassified crops if any
    if [ -d "./data/misclassified" ] && [ "$(ls -A ./data/misclassified 2>/dev/null)" ]; then
        echo "[train.sh] Syncing misclassified crops..."
        rsync -avz -e "ssh $SSH_OPTS" \
            ./data/misclassified/ \
            "${REMOTE_SSH_USER}@${REMOTE_SSH_HOST}:${REMOTE_DATA_DIR}/misclassified/"
    fi

    # 3. Run trainer on remote
    echo "[train.sh] Starting remote training..."
    ssh $SSH_OPTS "${REMOTE_SSH_USER}@${REMOTE_SSH_HOST}" \
        "docker run --rm --gpus all --shm-size=4g \
        --name ${CONTAINER_NAME} \
        -e MLFLOW_TRACKING_URI=${REMOTE_MLFLOW_URI} \
        -e DATASET_NAME=${DATASET_NAME} \
        -e N_TRIALS=${N_TRIALS} \
        -e N_EPOCHS_PER_TRIAL=${N_EPOCHS_PER_TRIAL} \
        -e LR_MIN=${LR_MIN} \
        -e LR_MAX=${LR_MAX} \
        -e MARGIN_MIN=${MARGIN_MIN} \
        -e MARGIN_MAX=${MARGIN_MAX} \
        -e MINING_CHOICES=${MINING_CHOICES} \
        -v ${REMOTE_CODE_DIR}:/app \
        -v ${REMOTE_DATA_DIR}:/app/data \
        -v ${REMOTE_CODE_DIR}/checkpoints:/app/checkpoints \
        -v ${REMOTE_CODE_DIR}/logs:/app/logs \
        trainer:latest python src/sweep_optuna.py"

    # 4. Pull checkpoints back so DVC can track them
    echo "[train.sh] Pulling checkpoints back..."
    rsync -avz -e "ssh $SSH_OPTS" \
        "${REMOTE_SSH_USER}@${REMOTE_SSH_HOST}:${REMOTE_CODE_DIR}/checkpoints/" \
        ./checkpoints/

    echo "[train.sh] Remote training complete."

# ─────────────────────────────────────────────────────────────────────────────
else
# ─────────────────────────────────────────────────────────────────────────────

    echo "[train.sh] MODE: LOCAL — running sweep_optuna.py directly"

    DATASET_NAME=${DATASET_NAME} \
    N_TRIALS=${N_TRIALS} \
    N_EPOCHS_PER_TRIAL=${N_EPOCHS_PER_TRIAL} \
    LR_MIN=${LR_MIN} \
    LR_MAX=${LR_MAX} \
    MARGIN_MIN=${MARGIN_MIN} \
    MARGIN_MAX=${MARGIN_MAX} \
    MINING_CHOICES=${MINING_CHOICES} \
    MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
    python src/sweep_optuna.py

    echo "[train.sh] Local training complete."

fi