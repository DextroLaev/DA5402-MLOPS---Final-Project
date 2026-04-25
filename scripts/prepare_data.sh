#!/bin/bash
set -euo pipefail

DATASET_NAME=${1:-lfw}

set -a
source "$(pwd)/.env"
set +a

docker run --rm \
    --network $COMPOSE_NETWORK \
    -e MLFLOW_TRACKING_URI=$REMOTE_MLFLOW_URI \
    -e DATASET_NAME=$DATASET_NAME \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/src:/app/src \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/results:/app/results \
    trainer:latest \
    python src/prepare_data.py  