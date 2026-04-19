import os
import cv2
import json
import time
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DB_PATH = "face_db.json"
THRESHOLD = 0.5

INFER_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def load_model(checkpoint):
    model = build_model().to(DEVICE)
    weights = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(weights["model_state_dict"])
    model.eval()
    print(f"Model loaded from {checkpoint}")
    return model


def get_embedding(model, face_bgr):
    """Run the encoder on a cropped face (BGR numpy array) and return a 1-D embedding."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = INFER_TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encoder(tensor)
    return emb.squeeze(0).cpu().numpy()


def load_db():
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, "r") as f:
        return json.load(f)


def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


def register_embedding(embedding, name):
    """Insert or update a named entry using a running weighted average."""
    db = load_db()

    for entry in db:
        if entry["label"] == name:
            old_emb = np.array(entry["embedding"])
            count   = entry.get("count", 1)
            entry["embedding"] = ((old_emb * count + embedding) / (count + 1)).tolist()
            entry["count"]     = count + 1
            save_db(db)
            print(f"Updated embedding for '{name}' (sample #{count + 1})")
            return

    db.append({"label": name, "embedding": embedding.tolist(), "count": 1})
    save_db(db)
    print(f"Registered new person: '{name}'")


def recognize_embedding(embedding):
    """Return the best-matching label, or 'Unknown' if below threshold."""
    db = load_db()
    if not db:
        return "Database is empty"

    query = embedding / np.linalg.norm(embedding)
    best_label, best_score = "Unknown", -1.0

    for entry in db:
        db_emb = np.array(entry["embedding"])
        db_emb = db_emb / np.linalg.norm(db_emb)
        score  = float(np.dot(query, db_emb))
        if score > best_score:
            best_score = score
            best_label = entry["label"]

    if best_score < THRESHOLD:
        return "Unknown"
    return f"{best_label} ({best_score:.2f})"


def run_camera(model, mode, name=None):
    """
    mode  : "register" | "recognize"
    name  : person's name (required when mode == "register")
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    CAPTURE_DURATION = 0.2          # seconds to hold steady before snap
    state      = "idle"
    start_time = 0.0
    processed  = False
    face_crop  = None

    print("Camera open. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        now   = time.time()

        # Draw boxes and track latest crop
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_crop = frame[y:y + h, x:x + w]

        # ── State machine ──────────────────────
        if state == "idle" and len(faces) > 0:
            state      = "capturing"
            start_time = now

        elif state == "capturing":
            if now - start_time >= CAPTURE_DURATION:
                state      = "done"
                start_time = now

        elif state == "done":
            cv2.putText(frame, "DONE", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

            if not processed and face_crop is not None:
                embedding = get_embedding(model, face_crop)

                if mode == "register":
                    register_embedding(embedding, name)
                    processed = True
                    cv2.putText(frame, "Registered! Closing...", (50, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow("Face Recognition", frame)
                    cv2.waitKey(1500)
                    break  # exit loop → closes camera

                else:
                    result = recognize_embedding(embedding)
                    print(f"Recognized: {result}")
                    cv2.putText(frame, result, (50, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow("Face Recognition", frame)
                    processed = True
                    time.sleep(1.0)  # 1 sec break before next recognition

            if now - start_time >= CAPTURE_DURATION:
                state     = "idle"
                processed = False

        # ───────────────────────────────────────
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")

    model = load_model("./checkpoints/best_model.pth")

    print("\n1. Register a person")
    print("2. Recognize a person")
    choice = input("Select (1/2): ").strip()

    if choice == "1":
        person_name = input("Enter the person's name: ").strip()
        if not person_name:
            print("Name cannot be empty.")
        else:
            run_camera(model, mode="register", name=person_name)

    elif choice == "2":
        run_camera(model, mode="recognize")

    else:
        print("Invalid choice.")