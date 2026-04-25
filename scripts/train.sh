#!/bin/bash
set -euo pipefail

DATASET_NAME=${1:-lfw}

# Load .env
set -a
source "$(pwd)/.env"
set +a

# Read params from params.yaml
N_TRIALS=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['train']['n_trials'])")
N_EPOCHS_PER_TRIAL=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['train']['n_epochs_per_trial'])")
LR_MIN=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['train']['lr_min'])")
LR_MAX=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['train']['lr_max'])")
MARGIN_MIN=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['train']['margin_min'])")
MARGIN_MAX=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['train']['margin_max'])")
MINING_CHOICES=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['train']['mining_choices'])")

if [ "$RUN_REMOTE" = "true" ]; then

    echo "[train.sh] MODE: REMOTE SERVER — ${REMOTE_SSH_HOST}"

    SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes"

    # 1. Sync code to server
    echo "[train.sh] Syncing code..."
    rsync -avz \
        --exclude='.git' \
        --exclude='.dvc/cache' \
        --exclude='data/' \
        --exclude='checkpoints/' \
        --exclude='monitoring/grafana/plugins/' \
        --exclude='monitoring/grafana/png/' \
        --exclude='monitoring/grafana/unified-search/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='results/' \
        --exclude='mlruns/' \
        --exclude='mlartifacts/' \
        -e "ssh $SSH_OPTS" \
        ./ "${REMOTE_SSH_USER}@${REMOTE_SSH_HOST}:${REMOTE_CODE_DIR}/"
    
    echo "[train.sh] Building trainer image on server..."
    ssh $SSH_OPTS "${REMOTE_SSH_USER}@${REMOTE_SSH_HOST}" \
        "cd ${REMOTE_CODE_DIR} && docker build -f Dockerfile.train -t trainer:latest ."


    # 2. Sync misclassified crops if any
    if [ -d "./data/misclassified" ] && [ "$(ls -A ./data/misclassified 2>/dev/null)" ]; then
        echo "[train.sh] Syncing misclassified crops..."
        rsync -avz -e "ssh $SSH_OPTS" \
            ./data/misclassified/ \
            "${REMOTE_SSH_USER}@${REMOTE_SSH_HOST}:${REMOTE_DATA_DIR}/misclassified/"
    fi

    # 3. Run trainer container on server
    echo "[train.sh] Starting training on server..."
    ssh $SSH_OPTS "${REMOTE_SSH_USER}@${REMOTE_SSH_HOST}" \
        "docker run --rm --gpus all --shm-size=4g \
        -e MLFLOW_TRACKING_URI=${REMOTE_MLFLOW_URI} \
        -e DATASET_NAME=${DATASET_NAME} \
        -e N_TRIALS=${N_TRIALS} \
        -e N_EPOCHS_PER_TRIAL=${N_EPOCHS_PER_TRIAL} \
        -e LR_MIN=${LR_MIN} \
        -e LR_MAX=${LR_MAX} \
        -e MARGIN_MIN=${MARGIN_MIN} \
        -e PYTHONUNBUFFERED=1 \
        -e MARGIN_MAX=${MARGIN_MAX} \
        -e MINING_CHOICES=${MINING_CHOICES} \
        -v ${REMOTE_CODE_DIR}:/app \
        -v ${REMOTE_DATA_DIR}:/app/data \
        -v $(pwd)/results:/app/results \
        trainer:latest python src/sweep_optuna.py"

    # 4. Pull checkpoints back
    echo "[train.sh] Pulling checkpoints back..."
    # sudo chown -R $(id -u):$(id -g) ./checkpoints
    rsync -avz -e "ssh $SSH_OPTS" \
        --include="*.json" \
        --include="*.txt" \
        --exclude="*" \
        "${REMOTE_SSH_USER}@${REMOTE_SSH_HOST}:${REMOTE_CODE_DIR}/checkpoints/" \
        ./checkpoints/

else

    echo "[train.sh] MODE: LOCAL DOCKER"

    docker run --rm \
        --gpus all \
        --shm-size=2g \
        --network ${COMPOSE_NETWORK} \
        -e PYTHONUNBUFFERED=1 \
        -e MLFLOW_TRACKING_URI=${REMOTE_MLFLOW_URI} \
        -e TORCH_HOME=/app/.cache/torch \
        -e DATASET_NAME=${DATASET_NAME} \
        -e N_TRIALS=${N_TRIALS} \
        -e N_EPOCHS_PER_TRIAL=${N_EPOCHS_PER_TRIAL} \
        -e LR_MIN=${LR_MIN} \
        -e LR_MAX=${LR_MAX} \
        -e PYTHONUNBUFFERED=1 \
        -e MARGIN_MIN=${MARGIN_MIN} \
        -e MARGIN_MAX=${MARGIN_MAX} \
        -e MINING_CHOICES=${MINING_CHOICES} \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/checkpoints:/app/checkpoints \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/src:/app/src \
        -v $(pwd)/params.yaml:/app/params.yaml \
        -v $(pwd)/results:/app/results \
        -v $HOME/.cache/torch:/app/.cache/torch \
        trainer:latest \
        python src/sweep_optuna.py

fi