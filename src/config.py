import os

# path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "lfw")           
PAIRS_FILE = os.path.join(DATA_DIR, "pairs.txt")  
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# image
IMG_SIZE = 105
IMG_CHANNELS = 3

# model
EMBEDDING_DIM = 128
DROPOUT_RATE = 0.3

def _env(name, default, cast=float):
    v = os.environ.get(name)
    return cast(v) if v is not None else default

# training
SEED = _env("SEED", 42, int)
BATCH_SIZE = _env("BATCH_SIZE", 64, int)
NUM_EPOCHS = _env("NUM_EPOCHS", 30, int)
LEARNING_RATE = _env("LEARNING_RATE", 1e-3, float)
WEIGHT_DECAY = _env("WEIGHT_DECAY", 1e-4, float)
LR_STEP_SIZE = _env("LR_STEP_SIZE", 10, int)
LR_GAMMA = _env("LR_GAMMA", 0.5, float)

# Two-phase training
WARMUP_EPOCHS = _env("WARMUP_EPOCHS", 5, int)
FINETUNE_LR = _env("FINETUNE_LR", 1e-4, float)
BACKBONE_LR = _env("BACKBONE_LR", 1e-5, float)

# triplet loss
MARGIN = _env("MARGIN", 0.5, float)
TRIPLET_MINING = _env("TRIPLET_MINING", "semi", str)
MIN_ACTIVE_TRIPLETS = _env("MIN_ACTIVE_TRIPLETS", 1, int)

# Train / val / test split ratios (applied to LFW identity map)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


# augmentation
USE_AUGMENTATION = True
HORIZONTAL_FLIP_P = 0.5
COLOR_JITTER = dict(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
RANDOM_ROTATION_DEG = 10

# inference
THRESHOLD = 0.8

NUM_WORKERS = 4
PIN_MEMORY = True

# ─── Logging / checkpointing ──────────────────────────────────────────────────
SAVE_EVERY_N_EPOCHS = 5
BEST_CKPT_NAME      = "best_model.pth"
LAST_CKPT_NAME      = "last_model.pth"