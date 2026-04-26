"""
prepare_data.py — Data preparation stage.

Run by:
  - DVC:    `dvc repro prepare_data`   (locally, for reproducibility)
  - Airflow: as the `prepare_data` task before `train_model`

What it does:
  1. Validates the LFW dataset (schema check, min-images-per-identity)
  2. Merges misclassified crops (same logic as dataloader, but explicit + logged)
  3. Computes and saves baseline statistics (mean, std, per-split counts)
     — these are used later for data drift detection
  4. Writes data/manifest.json so downstream stages know exact split contents
  5. Writes data/data_metrics.json for DVC metrics tracking
"""

import json
import os
import sys
import random
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Allow running from repo root or from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import config
from dataloader import scan_lfw, split_identities, merge_misclassified_into_train

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", config.TRAIN_RATIO))
VAL_RATIO = float(os.environ.get("VAL_RATIO",config.VAL_RATIO))
SEED = int(os.environ.get("SEED",config.SEED))
MIN_IMAGES = int(os.environ.get("MIN_IMAGES_PER_IDENTITY", "2"))

MANIFEST_PATH = os.path.join(config.DATA_DIR, f"{config.DATASET_NAME}_manifest.json")
BASELINE_PATH = os.path.join(config.DATA_DIR, f"{config.DATASET_NAME}_baseline_stats.json")
DATA_METRICS_PATH = os.path.join(config.DATA_DIR, f"{config.DATASET_NAME}_data_metrics.json")

_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
])


# ── Step 1: Validate dataset ──────────────────────────────────────────────────
def validate_dataset(raw_dir: str, min_images: int) -> dict:
    """
    Check that the dataset directory exists and has the expected structure.
    Returns identity_map if valid, raises ValueError if not.
    """
    print(f"[prepare] Validating dataset at: {raw_dir}")

    if not os.path.exists(raw_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {raw_dir}\n"
            f"Run: dvc pull  (to fetch tracked data from remote)\n"
            f"  or download LFW manually into data/{config.DATASET_NAME}/"
        )

    identity_map = scan_lfw(raw_dir, min_images=min_images)

    if len(identity_map) == 0:
        raise ValueError(
            f"No valid identities found in {raw_dir}. "
            f"Each person must have at least {min_images} images."
        )

    # Check for corrupted images (sample 5% of images)
    all_paths = [p for paths in identity_map.values() for p in paths]
    sample = random.Random(SEED).sample(all_paths, max(1, len(all_paths) // 20))
    corrupt = []
    for path in sample:
        try:
            Image.open(path).verify()
        except Exception as e:
            corrupt.append((path, str(e)))

    if corrupt:
        print(f"[prepare] WARNING: {len(corrupt)} potentially corrupt images found:")
        for path, err in corrupt[:5]:
            print(f"  {path}: {err}")

    print(f"[prepare] Validation passed: {len(identity_map)} identities, "
          f"{len(all_paths)} images, {len(corrupt)} suspect files")

    return identity_map


# ── Step 2: Compute baseline statistics ───────────────────────────────────────
def compute_baseline_stats(identity_map: dict, split_name: str, sample_size: int = 500) -> dict:
    """
    Compute per-split pixel statistics for drift detection.
    Samples up to `sample_size` images — full scan would be too slow.
    
    These baselines are saved to data/baseline_stats.json and compared
    against incoming data distributions during monitoring.
    """
    all_paths = [p for paths in identity_map.values() for p in paths]
    sampled   = random.Random(SEED).sample(all_paths, min(sample_size, len(all_paths)))

    means, stds = [], []
    for path in sampled:
        try:
            tensor = _TRANSFORM(Image.open(path).convert("RGB"))
            means.append(tensor.mean(dim=[1, 2]).numpy().tolist())   # [R, G, B]
            stds.append(tensor.std(dim=[1, 2]).numpy().tolist())
        except Exception:
            continue

    if not means:
        return {}

    means_arr = np.array(means)   # (N, 3)
    stds_arr  = np.array(stds)

    return {
        "split":            split_name,
        "n_identities":     len(identity_map),
        "n_images":         len(all_paths),
        "n_sampled":        len(means),
        "pixel_mean_rgb":   means_arr.mean(axis=0).tolist(),
        "pixel_std_rgb":    stds_arr.mean(axis=0).tolist(),
        "pixel_mean_var":   means_arr.var(axis=0).tolist(),   # variance across samples
    }


# ── Step 3: Write manifest ────────────────────────────────────────────────────
def write_manifest(train_map, val_map, test_map, misclassified_counts: dict) -> dict:
    manifest = {
        "dataset":               config.DATASET_NAME,
        "seed":                  SEED,
        "train_ratio":           TRAIN_RATIO,
        "val_ratio":             VAL_RATIO,
        "train_identities":      len(train_map),
        "val_identities":        len(val_map),
        "test_identities":       len(test_map),
        "train_images":          sum(len(v) for v in train_map.values()),
        "val_images":            sum(len(v) for v in val_map.values()),
        "test_images":           sum(len(v) for v in test_map.values()),
        "misclassified_merged":  misclassified_counts,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[prepare] Manifest written to {MANIFEST_PATH}")
    return manifest


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # 1. Validate
    identity_map = validate_dataset(config.RAW_DIR, MIN_IMAGES)

    # 2. Split (identity-level, same as dataloader.py)
    train_map, val_map, test_map = split_identities(
        identity_map, TRAIN_RATIO, VAL_RATIO, SEED
    )
    print(f"[prepare] Split: train={len(train_map)} val={len(val_map)} test={len(test_map)}")

    # 3. Merge misclassified crops into train only
    misclassified_path = os.path.join(config.DATA_DIR, "misclassified")
    before = len(train_map)
    train_map = merge_misclassified_into_train(train_map, misclassified_path)

    # Count how many crops were added per identity
    misc_counts = {}
    if os.path.exists(misclassified_path):
        for person in os.listdir(misclassified_path):
            pdir = os.path.join(misclassified_path, person)
            if os.path.isdir(pdir):
                misc_counts[person] = len([
                    f for f in os.listdir(pdir) if f.lower().endswith(".jpg")
                ])

    # 4. Compute baseline statistics (drift detection baselines)
    print("[prepare] Computing baseline statistics (this may take a moment)...")
    baseline = {
        "train": compute_baseline_stats(train_map, "train"),
        "val":   compute_baseline_stats(val_map,   "val"),
        "test":  compute_baseline_stats(test_map,  "test"),
    }
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"[prepare] Baseline stats written to {BASELINE_PATH}")

    # 5. Write manifest
    manifest = write_manifest(train_map, val_map, test_map, misc_counts)

    # 6. Write DVC metrics (lightweight JSON for `dvc metrics show`)
    data_metrics = {
        "train_identities": manifest["train_identities"],
        "val_identities":   manifest["val_identities"],
        "test_identities":  manifest["test_identities"],
        "train_images":     manifest["train_images"],
        "val_images":       manifest["val_images"],
        "misclassified_merged": sum(misc_counts.values()),
    }
    with open(DATA_METRICS_PATH, "w") as f:
        json.dump(data_metrics, f, indent=2)

    print("\n[prepare] ✓ Data preparation complete.")
    print(f"  Train : {manifest['train_identities']} identities / {manifest['train_images']} images")
    print(f"  Val   : {manifest['val_identities']} identities / {manifest['val_images']} images")
    print(f"  Test  : {manifest['test_identities']} identities / {manifest['test_images']} images")
    print(f"  Misc  : {sum(misc_counts.values())} crops merged across {len(misc_counts)} identities")


if __name__ == "__main__":
    main()