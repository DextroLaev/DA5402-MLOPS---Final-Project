"""
tests/conftest.py — Shared pytest fixtures and configuration.
"""

import os
import sys
import pytest
import torch

# Ensure src/ is always on the path for all test modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Set test environment variables before any imports
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///test_mlflow.db")
os.environ.setdefault("DATASET_NAME", "lfw")


def pytest_configure(config):
    """Register custom markers so pytest doesn't warn about unknown ones."""
    config.addinivalue_line("markers", "slow: mark test as slow (skipped in fast mode)")
    config.addinivalue_line("markers", "gpu: mark test as requiring a GPU")


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU tests when no CUDA is available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="No CUDA GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)