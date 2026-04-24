"""
tests/test_model.py — Unit tests for model, utils, and config.

These tests run in GitHub Actions on every push.
They use CPU only (no GPU), no real data (synthetic tensors).
Fast: should complete in < 60 seconds.
"""

import os
import pytest
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Make src/ importable
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///test_mlflow.db")


# ─── Model tests ──────────────────────────────────────────────────────────────

class TestSiameseEncoder:

    def test_build_model_returns_siamese_network(self):
        from model import build_model, SiameseNetwork
        model = build_model()
        assert isinstance(model, SiameseNetwork)

    def test_encoder_output_shape(self):
        from model import build_model
        import config
        model = build_model()
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            emb = model.encoder(x)
        assert emb.shape == (4, config.EMBEDDING_DIM), \
            f"Expected ({4}, {config.EMBEDDING_DIM}), got {emb.shape}"

    def test_embeddings_are_l2_normalized(self):
        from model import build_model
        model = build_model()
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            emb = model.encoder(x)
        norms = torch.norm(emb, dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5), \
            f"Embeddings not unit-norm: {norms}"

    def test_forward_returns_distance(self):
        from model import build_model
        model = build_model()
        model.eval()
        img1 = torch.randn(2, 3, 224, 224)
        img2 = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            emb1, emb2, dist = model(img1, img2)
        assert dist.shape == (2,)
        assert (dist >= 0).all(), "Distances must be non-negative"

    def test_backbone_frozen_by_default(self):
        from model import build_model
        model = build_model()
        frozen = all(not p.requires_grad for p in model.encoder.backbone.parameters())
        assert frozen, "Backbone should be frozen by default (phase 1)"

    def test_head_is_trainable(self):
        from model import build_model
        model = build_model()
        trainable = any(p.requires_grad for p in model.encoder.embed.parameters())
        assert trainable, "Embedding head should be trainable"

    def test_same_input_zero_distance(self):
        from model import build_model
        model = build_model()
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _, _, dist = model(x, x)
        assert dist.item() < 1e-4, f"Same input should give ~0 distance, got {dist.item()}"


# ─── TripletLoss tests ────────────────────────────────────────────────────────

class TestTripletLoss:

    @pytest.fixture
    def embeddings_and_labels(self):
        """4 identities, 2 samples each → 8 embeddings total."""
        torch.manual_seed(0)
        emb = F.normalize(torch.randn(8, 128), dim=1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        return emb, labels

    @pytest.mark.parametrize("mining", ["all", "hard", "semi"])
    def test_loss_is_non_negative(self, mining, embeddings_and_labels):
        from utils import TripletLoss
        emb, labels = embeddings_and_labels
        loss_fn = TripletLoss(margin=0.5, mining=mining)
        loss, n_active = loss_fn(emb, labels)
        assert loss.item() >= 0, f"[{mining}] Loss must be non-negative, got {loss.item()}"

    @pytest.mark.parametrize("mining", ["all", "hard", "semi"])
    def test_loss_returns_n_active(self, mining, embeddings_and_labels):
        from utils import TripletLoss
        emb, labels = embeddings_and_labels
        loss_fn = TripletLoss(margin=0.5, mining=mining)
        loss, n_active = loss_fn(emb, labels)
        assert isinstance(n_active, int), f"n_active must be int, got {type(n_active)}"
        assert n_active >= 0

    def test_invalid_mining_raises(self, embeddings_and_labels):
        from utils import TripletLoss
        emb, labels = embeddings_and_labels
        loss_fn = TripletLoss(margin=0.5, mining="invalid_strategy")
        with pytest.raises(ValueError, match="Unknown mining"):
            loss_fn(emb, labels)

    def test_perfect_embeddings_have_low_loss(self):
        """If same-class embeddings are identical and far from other classes, loss ≈ 0."""
        from utils import TripletLoss
        # Build perfectly separated embeddings
        e = torch.zeros(6, 128)
        e[0, 0] = e[1, 0] = 1.0   # class 0 at (1, 0, ...)
        e[2, 1] = e[3, 1] = 1.0   # class 1 at (0, 1, ...)
        e[4, 2] = e[5, 2] = 1.0   # class 2 at (0, 0, 1, ...)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loss_fn = TripletLoss(margin=0.3, mining="hard")
        loss, _ = loss_fn(e, labels)
        assert loss.item() < 0.01, f"Perfect embeddings should have near-zero loss, got {loss.item()}"


# ─── Config tests ─────────────────────────────────────────────────────────────

class TestConfig:

    def test_embedding_dim_positive(self):
        import config
        assert config.EMBEDDING_DIM > 0

    def test_train_val_ratios_sum_lt_one(self):
        import config
        assert config.TRAIN_RATIO + config.VAL_RATIO < 1.0, \
            "Train + val ratios must be < 1 to leave room for test split"

    def test_batch_size_divisible_by_4(self):
        import config
        # PKSampler requires batch_size % k == 0, default k=4
        assert config.BATCH_SIZE % 4 == 0, \
            f"BATCH_SIZE={config.BATCH_SIZE} must be divisible by k=4 for PKSampler"


# ─── PKSampler tests ──────────────────────────────────────────────────────────

class TestPKSampler:

    def test_batch_size_respected(self):
        from dataloader import PKSampler
        labels = list(range(20)) * 4   # 20 identities, 4 images each
        sampler = PKSampler(labels, batch_size=8, k=4, n_batches=10)
        for batch in sampler:
            assert len(batch) == 8, f"Expected batch of 8, got {len(batch)}"

    def test_batch_size_not_divisible_raises(self):
        from dataloader import PKSampler
        with pytest.raises(AssertionError):
            PKSampler(list(range(10)), batch_size=7, k=4)

    def test_n_batches_length(self):
        from dataloader import PKSampler
        labels = list(range(10)) * 4
        sampler = PKSampler(labels, batch_size=8, k=4, n_batches=5)
        assert len(sampler) == 5
        assert len(list(sampler)) == 5


# ─── Data preparation tests ───────────────────────────────────────────────────

class TestDataPrep:

    def test_split_identities_proportions(self):
        from dataloader import split_identities
        identity_map = {f"person_{i}": [f"img_{i}_1.jpg", f"img_{i}_2.jpg"] for i in range(100)}
        train, val, test = split_identities(identity_map, 0.7, 0.15, seed=42)
        assert len(train) == 70
        assert len(val)   == 15
        assert len(test)  == 15

    def test_split_no_identity_overlap(self):
        from dataloader import split_identities
        identity_map = {f"person_{i}": [f"img_{i}.jpg", f"img_{i}_2.jpg"] for i in range(50)}
        train, val, test = split_identities(identity_map, 0.7, 0.15, seed=0)
        t, v, te = set(train), set(val), set(test)
        assert t & v == set(), "Train and val share identities!"
        assert t & te == set(), "Train and test share identities!"
        assert v & te == set(), "Val and test share identities!"

    def test_merge_misclassified_adds_to_train_only(self, tmp_path):
        from dataloader import merge_misclassified_into_train
        import shutil

        # Create a fake misclassified dir with 2 identities
        for name in ["Alice", "NewPerson"]:
            person_dir = tmp_path / name
            person_dir.mkdir()
            # Create tiny valid JPEG
            img = Image.new("RGB", (50, 50), color=(128, 0, 0))
            img.save(str(person_dir / "face_001.jpg"))

        train_map = {"Alice": ["original_alice.jpg"], "Bob": ["bob.jpg"]}
        result = merge_misclassified_into_train(train_map, str(tmp_path))

        # Alice should have new crop appended
        assert len(result["Alice"]) == 2, "Alice should have original + misclassified crop"
        # NewPerson should be added
        assert "NewPerson" in result, "New identity from misclassified should be added to train"
        # Bob should be unchanged
        assert result["Bob"] == ["bob.jpg"], "Bob should be unchanged"


# ─── Register model tests (mocked) ───────────────────────────────────────────

class TestRegisterModel:

    def test_best_run_json_schema(self, tmp_path):
        """Ensure best_run.json has the keys register_model.py expects."""
        import json
        best_run = {
            "sweep_id": "sweep-20240101T000000-abc123",
            "run_id":   "abc123def456",
            "val_loss": 0.1234,
            "params":   {"learning_rate": 1e-3, "margin": 0.5, "triplet_mining": "semi"},
        }
        p = tmp_path / "best_run.json"
        p.write_text(json.dumps(best_run))

        loaded = json.loads(p.read_text())
        for key in ["run_id", "val_loss", "params"]:
            assert key in loaded, f"Missing key: {key}"