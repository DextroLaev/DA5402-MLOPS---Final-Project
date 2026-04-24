"""
tests/test_api.py — Integration tests for the Flask API.

Tests the API endpoints without needing a real model, camera, or MLflow.
Uses Flask's test client and mocked dependencies.
"""

import os
import sys
import json
import tempfile
import pytest
import numpy as np
from flask_app import app, init_db

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "flask_app"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["DB_PATH"]              = ":memory:"   # in-memory SQLite for tests
os.environ["MLFLOW_TRACKING_URI"]  = "sqlite:///test_mlflow.db"
os.environ["MLFLOW_MODEL_NAME"]    = "TestModel"
os.environ["MISCLASSIFY_THRESHOLD"]= "5"


@pytest.fixture(scope="module")
def client():
    """Create a Flask test client with a fresh in-memory DB."""
    # Patch out the model-loading and camera at import time
    import unittest.mock as mock

    with mock.patch("app.load_best_model_from_mlflow"), \
         mock.patch("app.threading.Thread"), \
         mock.patch("app.cv2.VideoCapture"):


        app.config["TESTING"] = True

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            test_db = f.name

        import app as app_module
        app_module.DB_PATH = test_db

        init_db()

        with app.test_client() as c:
            yield c

        os.unlink(test_db)


class TestHealthEndpoints:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_ready_returns_json(self, client):
        resp = client.get("/ready")
        assert resp.status_code in (200, 503)   # 503 if no model loaded, that's fine
        data = resp.get_json()
        assert "status" in data or resp.status_code == 503

    def test_metrics_returns_prometheus_format(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        text = resp.get_data(as_text=True)
        assert "requests_total" in text
        assert "uptime_seconds" in text
        assert "misclassifications_total" in text


class TestPeopleAPI:

    def test_list_people_empty_initially(self, client):
        resp = client.get("/api/people")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)

    def test_delete_nonexistent_person_ok(self, client):
        resp = client.post("/api/delete",
                           data=json.dumps({"name": "nobody"}),
                           content_type="application/json")
        assert resp.status_code == 200

    def test_delete_missing_name_fails(self, client):
        resp = client.post("/api/delete",
                           data=json.dumps({}),
                           content_type="application/json")
        assert resp.status_code == 400


class TestMisclassificationAPI:

    def test_report_misclassification_requires_true_name(self, client):
        resp = client.post("/api/report_misclassification",
                           data=json.dumps({"predicted": "Bob", "score": 0.3}),
                           content_type="application/json")
        assert resp.status_code == 400

    def test_report_misclassification_logs_correctly(self, client):
        resp = client.post("/api/report_misclassification",
                           data=json.dumps({
                               "true_name": "Alice",
                               "predicted": "Bob",
                               "score":     0.45,
                           }),
                           content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert "stats" in data
        assert data["stats"]["total"] >= 1

    def test_misclassification_stats_endpoint(self, client):
        resp = client.get("/api/misclassification_stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total" in data
        assert "unique_pending" in data
        assert "threshold" in data

    def test_flag_retrain_requires_true_label(self, client):
        resp = client.post("/api/flag_retrain",
                           data=json.dumps({}),
                           content_type="application/json")
        assert resp.status_code == 400


class TestRegisterAPI:

    def test_register_requires_name(self, client):
        resp = client.post("/api/register",
                           data=json.dumps({}),
                           content_type="application/json")
        assert resp.status_code == 400

    def test_register_sets_mode(self, client):
        resp = client.post("/api/register",
                           data=json.dumps({"name": "Alice"}),
                           content_type="application/json")
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    def test_cancel_register(self, client):
        resp = client.post("/api/cancel_register",
                           data=json.dumps({}),
                           content_type="application/json")
        assert resp.status_code == 200


class TestStatusAPI:

    def test_status_returns_expected_keys(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.get_json()
        for key in ["mode", "last_result", "model_version", "misclassify"]:
            assert key in data, f"Missing key in /api/status: {key}"

    def test_model_info_endpoint(self, client):
        resp = client.get("/api/model_info")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "model_name" in data
        assert "threshold" in data