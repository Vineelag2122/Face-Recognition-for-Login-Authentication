import base64
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import face_recognition
import mediapipe as mp
import numpy as np
from flask import Flask, flash, redirect, render_template, request, session, url_for
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_tasks_vision
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "users.db"
MODEL_DIR = BASE_DIR / "models"
PHONE_MODEL_PATH = MODEL_DIR / "efficientdet_lite0.tflite"
PHONE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
MATCH_TOLERANCE = 0.5
FACE_CASCADE = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))

app = Flask(__name__)
app.config["SECRET_KEY"] = "replace-with-a-secure-secret-key"

PHONE_DETECTOR = None
PHONE_DETECTOR_READY = False
PHONE_DETECTOR_INIT_DONE = False


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            image_path TEXT NOT NULL,
            face_encoding BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute("PRAGMA table_info(users)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    if "password" not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN password TEXT NOT NULL DEFAULT ''")

    conn.commit()
    conn.close()


def decode_image_payload(image_payload: str | None):
    if not image_payload:
        return None, "Please capture an image from the camera."

    if "," in image_payload:
        image_payload = image_payload.split(",", 1)[1]

    try:
        return base64.b64decode(image_payload), None
    except (ValueError, TypeError):
        return None, "Captured image data is invalid."


def _initialize_phone_detector() -> None:
    global PHONE_DETECTOR, PHONE_DETECTOR_READY, PHONE_DETECTOR_INIT_DONE

    if PHONE_DETECTOR_INIT_DONE:
        return

    PHONE_DETECTOR_INIT_DONE = True

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if not PHONE_MODEL_PATH.exists():
            urlretrieve(PHONE_MODEL_URL, PHONE_MODEL_PATH)

        base_options = mp_tasks_python.BaseOptions(model_asset_path=str(PHONE_MODEL_PATH))
        options = mp_tasks_vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=mp_tasks_vision.RunningMode.IMAGE,
            max_results=5,
            score_threshold=0.35,
            category_allowlist=["cell phone"],
        )

        PHONE_DETECTOR = mp_tasks_vision.ObjectDetector.create_from_options(options)
        PHONE_DETECTOR_READY = True
    except Exception:
        # Keep fallback heuristic available even if model download/init fails.
        PHONE_DETECTOR = None
        PHONE_DETECTOR_READY = False


def _detect_phone_with_model(img_bgr: np.ndarray) -> bool:
    _initialize_phone_detector()

    if not PHONE_DETECTOR_READY or PHONE_DETECTOR is None:
        return False

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = PHONE_DETECTOR.detect(mp_image)

    return bool(detection_result.detections)


def _detect_phone_with_heuristics(img_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 180)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_h, frame_w = gray.shape
    frame_area = float(frame_h * frame_w)
    frame_center = np.array([frame_w / 2.0, frame_h / 2.0], dtype=np.float32)
    frame_diag = float((frame_w**2 + frame_h**2) ** 0.5)

    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    suspicious_rectangles = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.012 or area > frame_area * 0.75:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if w < 35 or h < 35:
            continue

        aspect = w / float(max(1, h))
        aspect_ok = (0.35 <= aspect <= 0.85) or (1.2 <= aspect <= 2.4)
        if not aspect_ok:
            continue

        rectangularity = area / float(max(1, w * h))
        if rectangularity < 0.70:
            continue

        roi_gray = gray[y : y + h, x : x + w]
        roi_edges = edges[y : y + h, x : x + w]
        if roi_gray.size == 0 or roi_edges.size == 0:
            continue

        edge_density = float(np.count_nonzero(roi_edges)) / float(roi_edges.size)
        brightness_std = float(np.std(roi_gray))

        rect_center = np.array([x + (w / 2.0), y + (h / 2.0)], dtype=np.float32)
        is_centered = np.linalg.norm(rect_center - frame_center) < (0.45 * frame_diag)

        overlaps_face = False
        for fx, fy, fw, fh in faces:
            no_overlap = (x + w < fx) or (x > fx + fw) or (y + h < fy) or (y > fy + fh)
            if not no_overlap:
                overlaps_face = True
                break

        if edge_density > 0.07 and brightness_std > 30.0 and (is_centered or overlaps_face):
            suspicious_rectangles += 1

        if suspicious_rectangles >= 1:
            return True

    return False


def detect_phone_like_object_from_bytes(file_bytes: bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return False, "Could not read image for anti-spoof validation."

    if _detect_phone_with_model(img_bgr):
        return True, None

    if _detect_phone_with_heuristics(img_bgr):
        return True, None

    return False, None


def detect_phone_like_object_from_bytes(file_bytes: bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return False, "Could not read image for anti-spoof validation."

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 180)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_h, frame_w = gray.shape
    frame_area = float(frame_h * frame_w)
    frame_center = np.array([frame_w / 2.0, frame_h / 2.0], dtype=np.float32)
    frame_diag = float((frame_w**2 + frame_h**2) ** 0.5)

    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.015 or area > frame_area * 0.65:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if w < 40 or h < 40:
            continue

        aspect = w / float(max(1, h))
        aspect_ok = (0.35 <= aspect <= 0.8) or (1.25 <= aspect <= 2.2)
        if not aspect_ok:
            continue

        rectangularity = area / float(max(1, w * h))
        if rectangularity < 0.72:
            continue

        roi_gray = gray[y : y + h, x : x + w]
        roi_edges = edges[y : y + h, x : x + w]
        if roi_gray.size == 0 or roi_edges.size == 0:
            continue

        edge_density = float(np.count_nonzero(roi_edges)) / float(roi_edges.size)
        brightness_std = float(np.std(roi_gray))

        rect_center = np.array([x + (w / 2.0), y + (h / 2.0)], dtype=np.float32)
        is_centered = np.linalg.norm(rect_center - frame_center) < (0.38 * frame_diag)

        overlaps_face = False
        for fx, fy, fw, fh in faces:
            no_overlap = (x + w < fx) or (x > fx + fw) or (y + h < fy) or (y > fy + fh)
            if not no_overlap:
                overlaps_face = True
                break

        # Heuristic: phone-like rectangles typically have sharp edges and screen contrast
        if edge_density > 0.08 and brightness_std > 35.0 and (is_centered or overlaps_face):
            return True, None

    return False, None


def get_face_encoding_from_bytes(file_bytes: bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return None, "Could not read image. Capture a valid camera frame."

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(img_rgb)

    if len(encodings) == 0:
        return None, "No face found in the image."

    if len(encodings) > 1:
        return None, "Multiple faces detected. Upload an image with exactly one face."

    return encodings[0], None


def create_user(username: str, password: str, encoding: np.ndarray) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        password_hash = generate_password_hash(password)
        cursor.execute(
            """
            INSERT INTO users (username, password, image_path, face_encoding, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                username,
                password_hash,
                "camera-stream",
                pickle.dumps(encoding),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_user_encoding(username: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT face_encoding FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return pickle.loads(row[0])


def verify_user_password(username: str, password: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return False

    return check_password_hash(row[0], password)


def update_user_password(username: str, new_password: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        password_hash = generate_password_hash(new_password)
        cursor.execute("UPDATE users SET password = ? WHERE username = ?", (password_hash, username))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def update_user_face_profile(username: str, encoding: np.ndarray) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "UPDATE users SET face_encoding = ? WHERE username = ?",
            (pickle.dumps(encoding), username),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def register_user_flow(endpoint_name: str):
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    image_payload = request.form.get("face_image_data", "").strip()

    if not username:
        flash("Username is required.", "danger")
        return redirect(url_for(endpoint_name))

    if not password or len(password) < 6:
        flash("Password must be at least 6 characters.", "danger")
        return redirect(url_for(endpoint_name))

    file_bytes, error = decode_image_payload(image_payload)
    if error:
        flash(error, "danger")
        return redirect(url_for(endpoint_name))

    encoding, error = get_face_encoding_from_bytes(file_bytes)
    if error:
        flash(error, "danger")
        return redirect(url_for(endpoint_name))

    if create_user(username, password, encoding):
        flash("Registration successful! You can now log in.", "success")
        return redirect(url_for("login"))

    flash("Username already exists. Use a different username.", "warning")
    return redirect(url_for(endpoint_name))


init_db()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        return register_user_flow(endpoint_name="register")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        image_payload = request.form.get("face_image_data", "").strip()

        if not username:
            flash("Username is required.", "danger")
            return redirect(url_for("login"))

        if not password:
            flash("Password is required.", "danger")
            return redirect(url_for("login"))

        if not verify_user_password(username, password):
            flash("Invalid username or password.", "danger")
            return redirect(url_for("login"))

        file_bytes, error = decode_image_payload(image_payload)
        if error:
            flash(error, "danger")
            return redirect(url_for("login"))

        phone_detected, phone_error = detect_phone_like_object_from_bytes(file_bytes)
        if phone_error:
            flash(phone_error, "danger")
            return redirect(url_for("login"))

        if phone_detected:
            flash("Access denied. Phone-like object detected in live frame. Remove it and try again.", "danger")
            return redirect(url_for("login"))

        stored_encoding = get_user_encoding(username)
        if stored_encoding is None:
            flash("User not found.", "danger")
            return redirect(url_for("login"))

        login_encoding, error = get_face_encoding_from_bytes(file_bytes)
        if error:
            flash(error, "danger")
            return redirect(url_for("login"))

        distance = face_recognition.face_distance([stored_encoding], login_encoding)[0]
        is_face_match = bool(distance <= MATCH_TOLERANCE)
        is_match = bool(is_face_match)

        if is_match:
            flash(f"Login successful for {username}.", "success")
            session["authenticated_user"] = username
            session["login_summary"] = {
                "is_face_match": is_face_match,
                "distance": f"{distance:.4f}",
                "logged_in_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            return redirect(url_for("dashboard"))
        else:
            if not is_face_match:
                flash("Access denied. Face does not match.", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    username = session.get("authenticated_user")
    if not username:
        flash("Please log in first to access your dashboard.", "warning")
        return redirect(url_for("login"))

    if request.method == "POST":
        form_action = request.form.get("form_action", "change_password").strip()

        if form_action == "change_face_profile":
            face_password = request.form.get("face_password", "").strip()
            image_payload = request.form.get("face_image_data", "").strip()

            if not face_password:
                flash("Enter current passcode to update face profile.", "danger")
                return redirect(url_for("dashboard"))

            if not verify_user_password(username, face_password):
                flash("Current passcode is incorrect.", "danger")
                return redirect(url_for("dashboard"))

            file_bytes, error = decode_image_payload(image_payload)
            if error:
                flash(error, "danger")
                return redirect(url_for("dashboard"))

            encoding, encoding_error = get_face_encoding_from_bytes(file_bytes)
            if encoding_error:
                flash(encoding_error, "danger")
                return redirect(url_for("dashboard"))

            if update_user_face_profile(username, encoding):
                flash("Face profile updated successfully.", "success")
            else:
                flash("Could not update face profile. Please try again.", "danger")

            return redirect(url_for("dashboard"))

        current_password = request.form.get("current_password", "").strip()
        new_password = request.form.get("new_password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not current_password or not new_password or not confirm_password:
            flash("All password fields are required.", "danger")
            return redirect(url_for("dashboard"))

        if not verify_user_password(username, current_password):
            flash("Current passcode is incorrect.", "danger")
            return redirect(url_for("dashboard"))

        if len(new_password) < 6:
            flash("New passcode must be at least 6 characters.", "danger")
            return redirect(url_for("dashboard"))

        if new_password != confirm_password:
            flash("New passcode and confirm passcode do not match.", "danger")
            return redirect(url_for("dashboard"))

        if current_password == new_password:
            flash("New passcode must be different from the current passcode.", "warning")
            return redirect(url_for("dashboard"))

        if update_user_password(username, new_password):
            flash("Passcode updated successfully.", "success")
        else:
            flash("Could not update passcode. Please try again.", "danger")

        return redirect(url_for("dashboard"))

    login_summary = session.get("login_summary", {})
    return render_template("dashboard.html", username=username, login_summary=login_summary)


@app.route("/logout")
def logout():
    session.pop("authenticated_user", None)
    session.pop("login_summary", None)
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
