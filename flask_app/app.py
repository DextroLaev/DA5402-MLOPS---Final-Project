import os
import cv2
import time
import signal
import sqlite3
import threading
import numpy as np
import torch
import torchvision.transforms as T
import requests as http_requests
from PIL import Image
from flask import Flask, Response, render_template, request, jsonify
import mlflow
import mlflow.pytorch
import time as _time
import datetime
from prometheus_client import Gauge,REGISTRY

# ── MediaPipe — neural-network face detection ─────────────────────────────────
# Replaces Haar Cascade: handles extreme angles, low light, partial occlusion.
import mediapipe as mp

_mp_face = mp.solutions.face_detection
recognitions_total        = 0
recognitions_today_total  = 0
_today_date               = None
_unique_today_recognized  = set()  
_registrations_today      = 0      
_today_date_reg           = None   
_today_date = None
latency_sum = 0.0
latency_count = 0

_stop = threading.Event()

def _get_or_create_gauge(name, description):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Gauge(name, description)

misclassification_count = _get_or_create_gauge(
    'face_misclassification_count',
    'Number of pending misclassified faces in DB'
)

def update_misclassification_metric():
    con = sqlite3.connect(DB_PATH)
    count = con.execute("SELECT COUNT(*) FROM misclassified_faces").fetchone()[0]
    con.close()
    misclassification_count.set(count)

def _record_recognition(latency_s: float, label: str):
    global recognitions_total, recognitions_today_total, _today_date, _unique_today_recognized,latency_sum,latency_count

    today = datetime.datetime.utcnow().date()
    if today != _today_date:
        recognitions_today_total = 0
        _unique_today_recognized = set()
        _today_date = today

    recognitions_total       += 1
    recognitions_today_total += 1
    _unique_today_recognized.add(label)
    latency_sum += latency_s
    latency_count += 1
 
def _record_registration():
    global _registrations_today, _today_date_reg

    import datetime
    today = datetime.datetime.utcnow().date()
    if today != _today_date_reg:
        _registrations_today = 0
        _today_date_reg = today

    _registrations_today += 1

def _people_registered():
    import sqlite3
    con = sqlite3.connect(DB_PATH)
    n = con.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
    con.close()
    return n

def _latency_histogram_lines():
    lines = []
    for le in _BUCKETS:
        le_s = "+Inf" if le == float("inf") else str(le)
        lines.append(f'recognition_latency_seconds_bucket{{le="{le_s}"}} {latency_count}')
    lines.append(f"recognition_latency_seconds_sum {latency_sum:.6f}")
    lines.append(f"recognition_latency_seconds_count {latency_count}")
    return "\n".join(lines)


_BUCKETS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, float("inf")]

def _handle_signal(sig, frame):
    print("\nShutting down…")
    _stop.set()
    if cap is not None:
        cap.release()
    os._exit(0)

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DB_PATH = os.environ.get("DB_PATH", "face_db.sqlite")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

# MLflow settings — override via env vars in docker-compose
MLFLOW_TRACKING_URI     = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_MODEL_NAME       = os.environ.get("MLFLOW_MODEL_NAME", "FaceRecognitionModel")

# Airflow REST API settings
AIRFLOW_BASE_URL        = os.environ.get("AIRFLOW_BASE_URL", "http://airflow-webserver:8080")
AIRFLOW_USERNAME        = os.environ.get("AIRFLOW_USERNAME", "airflow")
AIRFLOW_PASSWORD        = os.environ.get("AIRFLOW_PASSWORD", "airflow")
RETRAIN_DAG_ID          = os.environ.get("RETRAIN_DAG_ID", "face_retraining_dag")

# Misclassification threshold: trigger retraining after this many unique misclassified people
MISCLASSIFY_THRESHOLD   = int(os.environ.get("MISCLASSIFY_THRESHOLD", "2"))

# ── MediaPipe detector settings ───────────────────────────────────────────────
# model_selection=1  → full-range model (detects faces up to 5 m away, better
#                       at extreme angles/distances than the short-range model 0)
# min_detection_confidence → lower = detects more faces (inc. difficult ones)
#                             but may introduce false positives. 0.4 is a good
#                             balance for challenging real-world conditions.
MP_MODEL_SELECTION       = int(os.environ.get("MP_MODEL_SELECTION", "1"))
MP_MIN_DETECTION_CONF    = float(os.environ.get("MP_MIN_DETECTION_CONF", "0.4"))

# Padding fraction added around the raw bounding box so the crop fed to the
# embedding model contains forehead / chin — identical to what registration
# captures — avoiding a tight-crop vs loose-crop mismatch.
MP_BBOX_PAD              = float(os.environ.get("MP_BBOX_PAD", "0.20"))

INFER_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

app = Flask(__name__)

# ── Metrics counters (Prometheus-compatible) ──────────────────────────────────
request_count = 0
error_count   = 0
start_time    = time.time()

# ── Global state ──────────────────────────────────────────────────────────────
model          = None
model_version  = None   # tracks which MLflow model version is loaded
cap            = None
face_detector  = None   # MediaPipe FaceDetection instance
_model_lock    = threading.Lock()

g_state = {
    "mode":          "recognize",
    "reg_name":      "",
    "last_result":   "",
    "reg_status":    "",
    "verify_result": "",
    "hitl_pending":  False,
    "hitl_embedding": None,
    "hitl_face_crop": None,
    "hitl_predicted": "",
    "hitl_score":     0.0,
    '_last_face_crop':None,
}

# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            label     TEXT    NOT NULL UNIQUE,
            embedding BLOB    NOT NULL,
            count     INTEGER NOT NULL DEFAULT 1
        )
    """)
    # Table for logging misclassification events
    con.execute("""
        CREATE TABLE IF NOT EXISTS misclassifications (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            true_name     TEXT    NOT NULL,
            predicted     TEXT,
            score         REAL,
            corrected     INTEGER NOT NULL DEFAULT 0,
            timestamp     REAL    NOT NULL
        )
    """)
    # Table for misclassified face crops (raw bytes) used as retraining data
    con.execute("""
        CREATE TABLE IF NOT EXISTS misclassified_faces (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            true_name     TEXT    NOT NULL,
            face_bytes    BLOB    NOT NULL,
            timestamp     REAL    NOT NULL
        )
    """)
    # con.execute("""
    #     CREATE TABLE IF NOT EXISTS retrain_queue (
    #         id         INTEGER PRIMARY KEY AUTOINCREMENT,
    #         true_label TEXT    NOT NULL,
    #         embedding  BLOB    NOT NULL,
    #         flagged_at TEXT    DEFAULT (datetime('now'))
    #     )
    # """)
    con.commit()
    con.close()


# ─── Face DB helpers ──────────────────────────────────────────────────────────
def db_register(embedding: np.ndarray, name: str):
    emb_bytes = embedding.astype(np.float32).tobytes()
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT embedding, count FROM faces WHERE label=?", (name,)).fetchone()
    if row:
        old_emb = np.frombuffer(row[0], dtype=np.float32)
        count   = row[1]
        new_emb = ((old_emb * count + embedding) / (count + 1)).astype(np.float32)
        con.execute(
            "UPDATE faces SET embedding=?, count=? WHERE label=?",
            (new_emb.tobytes(), count + 1, name)
        )
    else:
        con.execute(
            "INSERT INTO faces (label, embedding, count) VALUES (?,?,1)",
            (name, emb_bytes)
        )
    con.commit()
    con.close()


def db_recognize(embedding: np.ndarray):
    con  = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT label, embedding FROM faces").fetchall()
    con.close()

    if not rows:
        return "No faces registered yet", -1.0

    query = embedding / (np.linalg.norm(embedding) + 1e-9)
    best_label, best_score = "Unknown", -1.0
    for label, emb_blob in rows:
        db_emb = np.frombuffer(emb_blob, dtype=np.float32)
        db_emb = db_emb / (np.linalg.norm(db_emb) + 1e-9)
        score  = float(np.dot(query, db_emb))
        if score > best_score:
            best_score, best_label = score, label

    if best_score < THRESHOLD:
        return "Unknown", best_score
    return best_label, best_score


def db_list_people():
    con  = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT label, count FROM faces ORDER BY label").fetchall()
    con.close()
    return [{"name": r[0], "samples": r[1]} for r in rows]


def db_delete_person(name: str):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM faces WHERE label=?", (name,))
    con.commit()
    con.close()


# ── Misclassification helpers ─────────────────────────────────────────────────
def log_misclassification(true_name: str, predicted: str, score: float,
                           face_bgr: np.ndarray = None):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO misclassifications (true_name, predicted, score, timestamp) "
        "VALUES (?,?,?,?)",
        (true_name, predicted, score, time.time())
    )

    if face_bgr is not None:
        ok, buf = cv2.imencode(".jpg", face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            con.execute(
                "INSERT INTO misclassified_faces (true_name, face_bytes, timestamp) "
                "VALUES (?,?,?)",
                (true_name, buf.tobytes(), time.time())
            )

    row = con.execute(
        "SELECT COUNT(DISTINCT true_name) FROM misclassifications WHERE corrected=0"
    ).fetchone()
    pending_count = row[0] if row else 0
    con.commit()
    con.close()

    print(f"[Misclassification] true={true_name}  pred={predicted}  "
          f"score={score:.3f}  pending_misclassified={pending_count}")

    if pending_count >= MISCLASSIFY_THRESHOLD:
        _trigger_retraining_dag()
    
    update_misclassification_metric()


def _get_airflow_token() -> str:
    """Exchange username/password for a JWT Bearer token (Airflow 3.x)."""
    resp = http_requests.post(
        f"{AIRFLOW_BASE_URL}/auth/token",
        json={"username": AIRFLOW_USERNAME, "password": AIRFLOW_PASSWORD},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _trigger_retraining_dag():
    url = f"{AIRFLOW_BASE_URL}/api/v2/dags/{RETRAIN_DAG_ID}/dagRuns"
    now = datetime.datetime.now(datetime.timezone.utc)
    logical_date = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    payload = {
        "dag_run_id":   f"misclassify_{int(now.timestamp())}",
        "logical_date": logical_date,
        "conf": {
            "triggered_by":       "misclassification_threshold",
            "threshold":          MISCLASSIFY_THRESHOLD,
            "timestamp":          now.timestamp(),
            "n_epochs_per_trial": 25,
        },
    }
    try:
        token = _get_airflow_token()                          # ← get JWT first
        resp = http_requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},    # ← Bearer, not Basic
            timeout=10,
        )
        if resp.status_code in (200, 201):
            print(f"[Airflow] Retraining DAG triggered: {resp.json().get('dag_run_id')}")
            _mark_misclassifications_triggered()
        else:
            print(f"[Airflow] Failed to trigger DAG: {resp.status_code} {resp.text}")
    except Exception as exc:
        print(f"[Airflow] Error triggering DAG: {exc}")


def _mark_misclassifications_triggered():
    con.execute("UPDATE misclassifications SET corrected=1 WHERE corrected=0")
    con.execute("DELETE FROM misclassified_faces")  # ← add this line
    con.commit()
    con.close()
    update_misclassification_metric()


def get_misclassification_stats():
    con = sqlite3.connect(DB_PATH)
    total = con.execute("SELECT COUNT(*) FROM misclassifications").fetchone()[0]
    pending = con.execute(
        "SELECT COUNT(*) FROM misclassifications WHERE corrected=0"
    ).fetchone()[0]
    con.close()
    return {"total": total, "unique_pending": pending, "threshold": MISCLASSIFY_THRESHOLD}


# ── MLflow model loading ──────────────────────────────────────────────────────
def load_best_model_from_mlflow():
    global model, model_version

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        prod_versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
        if prod_versions:
            mv = prod_versions[0]
        else:
            all_versions = client.get_latest_versions(
                MLFLOW_MODEL_NAME, stages=["None", "Staging", "Production"]
            )
            if not all_versions:
                raise RuntimeError(
                    f"No versions found for model '{MLFLOW_MODEL_NAME}' in MLflow registry"
                )
            mv = max(all_versions, key=lambda v: int(v.version))

        run_id  = mv.run_id
        version = mv.version

        print(f"[MLflow] Loading model '{MLFLOW_MODEL_NAME}' version={version} run_id={run_id}")

        model_uri = f"runs:/{run_id}/best_model"
        m = mlflow.pytorch.load_model(model_uri, map_location=DEVICE).to(DEVICE)
        m.eval()

        with _model_lock:
            model         = m
            model_version = version

        print(f"[MLflow] Model loaded — version {version}, device={DEVICE}")
        return version

    except Exception as exc:
        print(f"[MLflow] ERROR loading model: {exc}")
        raise


def _find_checkpoint(base_path: str):
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.endswith(".pth"):
                return os.path.join(root, f)
    return None


def _model_reload_loop():
    CHECK_INTERVAL = int(os.environ.get("MODEL_RELOAD_INTERVAL_SECS", "60"))
    while not _stop.is_set():
        time.sleep(CHECK_INTERVAL)
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient()
            prod_versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
            if not prod_versions:
                continue
            latest = prod_versions[0]
            if str(latest.version) != str(model_version):
                print(f"[MLflow] New Production model detected (v{latest.version}), reloading…")
                load_best_model_from_mlflow()
        except Exception as exc:
            print(f"[MLflow] Reload check failed: {exc}")


# ── Model inference ───────────────────────────────────────────────────────────
def get_embedding(face_bgr: np.ndarray) -> np.ndarray:
    rgb    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = INFER_TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    with _model_lock:
        with torch.no_grad():
            emb = model.encoder(tensor)
    return emb.squeeze(0).cpu().numpy()


# ── MediaPipe face detection helpers ─────────────────────────────────────────
def init_face_detector():
    """
    Create a MediaPipe FaceDetection instance.

    model_selection=1 uses the full-range model trained on a much wider variety
    of poses, distances and lighting than the short-range model (0). It is the
    right choice for a surveillance / door-entry style camera.
    """
    return _mp_face.FaceDetection(
        model_selection=MP_MODEL_SELECTION,
        min_detection_confidence=MP_MIN_DETECTION_CONF,
    )


def detect_faces_mp(detector, frame_bgr: np.ndarray):
    """
    Run MediaPipe face detection on a BGR frame.

    Returns a list of (x, y, w, h) pixel bounding boxes, padded by
    MP_BBOX_PAD so the crop includes enough facial context for the
    embedding model.

    MediaPipe expects RGB input; we convert here so callers can stay
    in BGR-land just like before.
    """
    h, w = frame_bgr.shape[:2]
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # process() is not thread-safe — the caller already serialises via gen_frames
    results = detector.process(rgb)

    boxes = []
    if not results.detections:
        return boxes

    for det in results.detections:
        bb   = det.location_data.relative_bounding_box
        # Raw normalised coords (can be slightly outside [0,1])
        rx, ry, rw, rh = bb.xmin, bb.ymin, bb.width, bb.height

        # Apply padding symmetrically around the raw box
        pad_x = rw * MP_BBOX_PAD
        pad_y = rh * MP_BBOX_PAD
        rx = rx - pad_x;  rw = rw + 2 * pad_x
        ry = ry - pad_y;  rh = rh + 2 * pad_y

        # Convert to pixel coords and clamp
        x1 = max(0, int(rx * w))
        y1 = max(0, int(ry * h))
        x2 = min(w, int((rx + rw) * w))
        y2 = min(h, int((ry + rh) * h))

        bw, bh = x2 - x1, y2 - y1
        if bw > 10 and bh > 10:          # discard degenerate boxes
            boxes.append((x1, y1, bw, bh))

    return boxes


# ── Video generator ───────────────────────────────────────────────────────────
def find_camera():
    for idx in range(6):
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            ret, _ = c.read()
            if ret:
                print(f"Camera found at index {idx}")
                return c
            c.release()
    print("ERROR: No working camera found (indices 0-5 tried)")
    return None


def gen_frames():
    global cap, face_detector

    cap = find_camera()
    if cap is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "No camera detected", (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 255), 2)
        _, buf = cv2.imencode(".jpg", blank)
        while not _stop.is_set():
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
            time.sleep(0.1)
        return

    # ── Initialise MediaPipe detector (once per stream) ───────────────────────
    # MediaPipe context managers keep internal state; we open it here and
    # close it when the generator exits so resources are properly released.
    face_detector = init_face_detector()

    reg_state = "idle"
    reg_start = 0.0
    reg_crop  = None
    HOLD_SECS = 1

    try:
        while not _stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            # ── Detect faces with MediaPipe ───────────────────────────────────
            faces = detect_faces_mp(face_detector, frame)
            now   = time.time()

            face_crop = None
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 180), 2)
                face_crop = frame[y:y + h, x:x + w]

            mode = g_state["mode"]
            g_state['_last_face_crop'] = face_crop

            # ── RECOGNITION mode ──────────────────────────────────────────────
            if mode == "recognize":
                reg_state = "idle"
                if face_crop is not None and model is not None:
                    _t0 = time.time()
                    emb   = get_embedding(face_crop)
                    label, score = db_recognize(emb)
                    _latency = _time.time() - _t0
                    if label == "Unknown":
                        g_state["last_result"] = "Unknown"
                        color = (60, 60, 255)
                    else:
                        g_state["last_result"] = f"{label}|{score:.2f}"
                        color = (0, 255, 180)

                    if label not in ("Unknown", "No faces registered yet"):
                        _record_recognition(_latency, label)   # ← only known faces
                    display = label if label == "Unknown" else f"{label}  {score:.0%}"
                    cv2.putText(frame, display, (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

                    if model_version:
                        cv2.putText(frame, f"Model v{model_version}",
                                    (20, frame.shape[0] - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                elif len(faces) == 0:
                    g_state["last_result"] = ""

            # ── REGISTER mode ─────────────────────────────────────────────────
            elif mode == "register":
                name = g_state["reg_name"]

                if reg_state == "idle" and len(faces) > 0:
                    reg_state = "capturing"
                    reg_start = now

                elif reg_state == "capturing":
                    remaining = HOLD_SECS - (now - reg_start)
                    cv2.putText(frame, f"Hold still… {remaining:.1f}s", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)
                    if face_crop is not None:
                        reg_crop = face_crop
                    if now - reg_start >= HOLD_SECS:
                        reg_state = "done"

                elif reg_state == "done":
                    if reg_crop is not None and model is not None and name:
                        emb = get_embedding(reg_crop)
                        db_register(emb, name)
                        _record_registration()

                        v_label, v_score = db_recognize(emb)
                        if v_label == name:
                            g_state["verify_result"] = f"Verified as {name}  ({v_score:.0%})"
                            g_state["reg_status"]    = "success"
                        else:
                            g_state["verify_result"] = f"Registered but recognized as {v_label}"
                            g_state["reg_status"]    = "warning"

                        cv2.putText(frame, "Registered!", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 180), 3, cv2.LINE_AA)

                        g_state["hitl_embedding"] = emb
                        g_state["hitl_pending"] = True
                        g_state["reg_status"] = "hitl_ready"
                        g_state["verify_result"] = ""

                    g_state["mode"] = "recognize"
                    g_state["reg_name"] = ""
                    reg_state = "idle"
                    reg_crop = None

            ret2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret2:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
    finally:
        # Always release MediaPipe resources cleanly
        face_detector.close()
        face_detector = None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "uptime_seconds": round(time.time() - start_time)})


@app.route("/ready")
def ready():
    return jsonify({
        "status":        "ready" if model is not None else "not_ready",
        "model":         MLFLOW_MODEL_NAME,
        "model_version": model_version,
    })


@app.route("/metrics")
def metrics():
    stats        = get_misclassification_stats()
    people       = _people_registered()
    model_loaded = 1 if model is not None else 0
 
    output = (
        "# HELP requests_total Total HTTP requests to Flask\n"
        "# TYPE requests_total counter\n"
        f"requests_total {request_count}\n"
 
        "# HELP errors_total Total API errors\n"
        "# TYPE errors_total counter\n"
        f"errors_total {error_count}\n"
 
        "# HELP uptime_seconds API uptime in seconds\n"
        "# TYPE uptime_seconds gauge\n"
        f"uptime_seconds {round(_time.time() - start_time)}\n"

        "# HELP model_loaded 1 if a model is loaded and ready, 0 otherwise\n"
        "# TYPE model_loaded gauge\n"
        f"model_loaded {model_loaded}\n"
 
        f"# HELP model_version_info Currently loaded MLflow model version\n"
        "# TYPE model_version_info gauge\n"
        f'model_version_info{{version="{model_version or "none"}"}} {model_loaded}\n'

        "# HELP people_registered_total Total unique people in the face database\n"
        "# TYPE people_registered_total gauge\n"
        f"people_registered_total {people}\n"
 
        "# HELP recognitions_total All-time successful face recognition events\n"
        "# TYPE recognitions_total counter\n"
        f"recognitions_total {recognitions_total}\n"
 
        "# HELP recognitions_today_total Successful recognitions since midnight UTC\n"
        "# TYPE recognitions_today_total gauge\n"
        f"recognitions_today_total {recognitions_today_total}\n"
 
        "# HELP recognition_latency_seconds Time from face crop to label returned\n"
        "# TYPE recognition_latency_seconds histogram\n"
        + _latency_histogram_lines() + "\n"
 
        "# HELP misclassifications_total Total misclassification events logged\n"
        "# TYPE misclassifications_total counter\n"
        f"misclassifications_total {stats['total']}\n"
 
        "# HELP misclassifications_pending Unique persons pending retraining trigger\n"
        "# TYPE misclassifications_pending gauge\n"
        f"misclassifications_pending {stats['unique_pending']}\n"
 
        "# HELP retraining_threshold Unique-person threshold to trigger retraining\n"
        "# TYPE retraining_threshold gauge\n"
        f"retraining_threshold {stats['threshold']}\n"
        "# HELP people_registered_total Total unique people in the face DB (all time)\n"
        "# TYPE people_registered_total gauge\n"
        f"people_registered_total {_people_registered()}\n"

        "# HELP registrations_today New unique persons registered since midnight UTC\n"
        "# TYPE registrations_today gauge\n"
        f"registrations_today {_registrations_today}\n"

        "# HELP unique_recognitions_today Unique persons recognised since midnight UTC\n"
        "# TYPE unique_recognitions_today gauge\n"
        f"unique_recognitions_today {len(_unique_today_recognized)}\n"
    )
    return output, 200, {"Content-Type": "text/plain; version=0.0.4"}


@app.route("/api/status")
def api_status():
    return jsonify({
        "mode":            g_state["mode"],
        "last_result":     g_state["last_result"],
        "reg_status":      g_state["reg_status"],
        "verify_result":   g_state["verify_result"],
        "model_version":   model_version,
        "misclassify":     get_misclassification_stats(),
        "hitl_pending":    g_state["hitl_pending"],
    })

@app.route("/api/test_recognition", methods=["POST"])
def api_test_recognition():
    """Run one inference on the most-recently seen face and return result."""
    crop = g_state.get("_last_face_crop")
    if crop is None or model is None:
        return jsonify({"ok": False, "error": "No face visible — stand in front of camera first"})

    emb = get_embedding(crop)
    label, score = db_recognize(emb)
    g_state["hitl_embedding"] = emb
    g_state["hitl_face_crop"] = crop.copy()
    g_state["hitl_predicted"] = label
    g_state["hitl_score"]     = float(score)
    return jsonify({"ok": True, "name": label, "score": round(float(score), 3)})

@app.route("/api/flag_retrain", methods=["POST"])
def api_flag_retrain():
    """User said recognition was wrong — save face crop with the correct label as a retraining sample."""
    data = request.get_json(force=True)
    true_label = (data.get("true_label") or "").strip()
    if not true_label:
        return jsonify({"ok": False, "error": "true_label is required"}), 400

    crop = g_state.get("hitl_face_crop")
    if crop is None:
        return jsonify({"ok": False, "error": "No face crop to flag — run Test Recognition first"}), 400

    predicted = g_state.get("hitl_predicted", "")
    score     = float(g_state.get("hitl_score", 0.0))

    log_misclassification(true_label, predicted, score, face_bgr=crop)

    try:
        out_dir = os.path.join("/app/data", "misclassified", true_label)
        os.makedirs(out_dir, exist_ok=True)
        fname   = f"{int(time.time()*1000)}.jpg"
        cv2.imwrite(os.path.join(out_dir, fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    except Exception as exc:
        print(f"[flag_retrain] disk save skipped: {exc}")

    g_state["hitl_pending"]   = False
    g_state["hitl_embedding"] = None
    g_state["hitl_face_crop"] = None
    g_state["hitl_predicted"] = ""
    g_state["hitl_score"]     = 0.0

    stats = get_misclassification_stats()
    return jsonify({"ok": True, "queued": true_label, "stats": stats})

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Name is required"}), 400

    g_state["mode"]          = "register"
    g_state["reg_name"]      = name
    g_state["reg_status"]    = "pending"
    g_state["verify_result"] = ""
    return jsonify({"ok": True})


@app.route("/api/cancel_register", methods=["POST"])
def api_cancel():
    g_state["mode"]     = "recognize"
    g_state["reg_name"] = ""
    return jsonify({"ok": True})


@app.route("/api/people")
def api_people():
    return jsonify(db_list_people())


@app.route("/api/delete", methods=["POST"])
def api_delete():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Name required"}), 400
    db_delete_person(name)
    return jsonify({"ok": True})


@app.route("/api/report_misclassification", methods=["POST"])
def api_report_misclassification():
    global request_count
    request_count += 1

    data      = request.get_json(force=True)
    true_name = (data.get("true_name") or "").strip()
    predicted = (data.get("predicted") or "").strip()
    score     = float(data.get("score", 0.0))

    if not true_name:
        return jsonify({"ok": False, "error": "true_name is required"}), 400

    log_misclassification(true_name, predicted, score)
    stats = get_misclassification_stats()
    return jsonify({
        "ok":                  True,
        "stats":               stats,
        "retraining_triggered": stats["unique_pending"] == 0,
    })


@app.route("/api/misclassification_stats")
def api_misclassification_stats():
    return jsonify(get_misclassification_stats())


@app.route("/api/trigger_retrain", methods=["POST"])
def api_trigger_retrain():
    """Manual override — trigger retraining immediately."""
    _trigger_retraining_dag()
    return jsonify({"ok": True, "message": "Retraining DAG trigger sent to Airflow"})


@app.route("/api/model_info")
def api_model_info():
    return jsonify({
        "model_name":    MLFLOW_MODEL_NAME,
        "model_version": model_version,
        "device":        str(DEVICE),
        "threshold":     THRESHOLD,
    })


# ── Boot ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()

    try:
        load_best_model_from_mlflow()
    except Exception as exc:
        print(f"[Boot] No model available yet ({exc}); serving without one.")

    reload_thread = threading.Thread(target=_model_reload_loop, daemon=True)
    reload_thread.start()

    app.run(host="0.0.0.0", port=2000, debug=False,
            threaded=True, use_reloader=False)