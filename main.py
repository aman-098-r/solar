"""
Solar Panel Cleaning - Python Backend
--------------------------------------
- FastAPI server on port 5001
- /stream          -> MJPEG webcam stream
- /predict         -> Capture a frame, run model, return score
- /start_inference -> Start continuous background inference loop
- /stop_inference  -> Stop background inference loop
- Scores are pushed to Firebase RTDB at /prediction

Model: MobileNetV3Large (Keras) pickled as solar_model.pkl
  - Input : 224x224 RGB, preprocessed with mobilenet_v3.preprocess_input
  - Output: dict with keys:
      'cls'  -> sigmoid probability (0-1): > 0.5 = Clean, <= 0.5 = Dusty
      'dust' -> dust severity score (1-10 scale)
"""

import io
import os
import sys
import warnings

# Silence TensorFlow noisy C++ warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings(
    "ignore",
    message=".*tf\\.placeholder is deprecated.*",
    category=UserWarning,
)

import pickle
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
FIREBASE_DB_URL = (
    "https://solar-cleaning-major-project-default-rtdb.firebaseio.com"
)
MODEL_DIR = Path(__file__).parent
MODEL_CANDIDATE_PATHS = [
    MODEL_DIR / "solar_model_patched.keras",
    MODEL_DIR / "solar_model.pkl",
    MODEL_DIR / "solar_model_backup.pkl",
]
WEBCAM_INDEX = 0          # Preferred webcam index
WEBCAM_BACKEND = cv2.CAP_DSHOW  # Stable backend on Windows
INFERENCE_INTERVAL = 5    # seconds between automatic predictions
JPEG_QUALITY = 70         # MJPEG quality 0-100
IMG_SIZE = (224, 224)     # MobileNetV3 input size
CAMERA_RECONNECT_DELAY = 1.0
CAPTURE_TARGET_FPS = 30.0
STREAM_TARGET_FPS = 16.0
STALE_FRAME_RECONNECT_SECONDS = 20.0
AUTO_START_INFERENCE = True

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
_model = None
_model_loaded = False
_model_lock = threading.Lock()
_model_error: Optional[str] = None


def load_model() -> Optional[object]:
    global _model, _model_loaded, _model_error
    if _model_loaded:
        return _model

    with _model_lock:
        if _model_loaded:
            return _model
        existing_candidates = [p for p in MODEL_CANDIDATE_PATHS if p.exists()]
        if existing_candidates:
            try:
                import tensorflow as tf  # noqa: F401 (validates TF is installed)
                tf.get_logger().setLevel("ERROR")
                try:
                    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
                except Exception:
                    pass
            except Exception as e:
                _model = None
                _model_loaded = False
                _model_error = f"TensorFlow import failed: {e}"
                print(f"[Model] [ERROR] {_model_error}")
                return _model

            for model_path in existing_candidates:
                try:
                    if model_path.suffix == ".keras":
                        _model = tf.keras.models.load_model(model_path, compile=False)
                    else:
                        if sys.version_info >= (3, 13):
                            # Known unstable on this runtime for pickle-serialized Keras models.
                            raise RuntimeError(
                                "Skipping pickle model on Python 3.13 (incompatible runtime crash risk)"
                            )
                        with open(model_path, "rb") as f:
                            _model = pickle.load(f)

                    _model_loaded = True
                    _model_error = None
                    print(f"[Model] Loaded model from {model_path}")
                    return _model
                except Exception as e:
                    _model = None
                    _model_loaded = False
                    _model_error = f"{model_path.name}: {e}"
                    print(f"[Model] [ERROR] Failed loading {model_path.name}: {e}")
        else:
            candidates = ", ".join(str(p.name) for p in MODEL_CANDIDATE_PATHS)
            print(f"[Model] [WARNING] Model file not found. Tried: {candidates}")
            _model_loaded = False
            _model_error = f"Model file not found. Tried: {candidates}"
        return _model


# ──────────────────────────────────────────────────────────────────────────────
# Webcam singleton
# ──────────────────────────────────────────────────────────────────────────────
class CameraManager:
    """Thread-safe singleton camera wrapper."""

    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_time = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._active_index: Optional[int] = None

    def _open_camera(self) -> bool:
        # Keep source stable: index 0 + DirectShow, then fallback to default backend.
        for backend in [WEBCAM_BACKEND, None]:
            if backend is None:
                cap = cv2.VideoCapture(WEBCAM_INDEX)
                backend_name = "default"
            else:
                cap = cv2.VideoCapture(WEBCAM_INDEX, backend)
                backend_name = str(backend)

            if not cap or not cap.isOpened():
                if cap:
                    cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, CAPTURE_TARGET_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                continue

            self._cap = cap
            self._active_index = WEBCAM_INDEX
            with self._lock:
                self._last_frame = frame
                self._last_frame_time = time.time()
            print(f"[Camera] Started on index {WEBCAM_INDEX} backend={backend_name}")
            return True

        self._cap = None
        self._active_index = None
        return False

    def start(self):
        if self._running:
            return
        if not self._open_camera():
            raise RuntimeError(
                "Cannot open any camera (tried indices 0-3). "
                "Close apps using the webcam, then restart backend."
            )
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        frame_interval = 1.0 / CAPTURE_TARGET_FPS
        while self._running:
            try:
                if self._cap is None or not self._cap.isOpened():
                    if self._cap:
                        self._cap.release()
                    self._cap = None
                    print("[Camera] Capture unavailable, attempting reconnect...")
                    if not self._open_camera():
                        time.sleep(CAMERA_RECONNECT_DELAY)
                        continue

                ret, frame = self._cap.read()
                if ret and frame is not None:
                    now = time.time()
                    with self._lock:
                        self._last_frame = frame
                        self._last_frame_time = now
                else:
                    # Keep camera alive; transient read failures are common on Windows.
                    # Avoid aggressive reconnects that power-cycle webcam.
                    time.sleep(0.01)

                # If the frame stream goes stale for too long, recover camera once.
                if self._last_frame_time and (
                    (time.time() - self._last_frame_time) > STALE_FRAME_RECONNECT_SECONDS
                ):
                    print("[Camera] No fresh frame for 20s, reconnecting camera...")
                    if self._cap:
                        self._cap.release()
                    self._cap = None
                    time.sleep(CAMERA_RECONNECT_DELAY)
                    continue

            except Exception as e:
                print(f"[Camera] Capture loop error: {e}")
                if self._cap:
                    self._cap.release()
                self._cap = None
                time.sleep(CAMERA_RECONNECT_DELAY)
                continue

            time.sleep(frame_interval)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._last_frame.copy() if self._last_frame is not None else None

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
        self._cap = None
        self._active_index = None
        self._last_frame_time = 0.0


camera = CameraManager()


# ──────────────────────────────────────────────────────────────────────────────
# Startup / Shutdown (lifespan style, backward compatible)
# ──────────────────────────────────────────────────────────────────────────────
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app):
    print("🚀 Starting backend...")

    # Start camera
    try:
        camera.start()
        print("✅ Camera started")
    except Exception as e:
        print(f"❌ Camera error: {e}")

    # Load model in background so camera stream is available immediately.
    def _warm_model():
        try:
            model = load_model()
            if model is not None:
                print("✅ Model loaded")
                if AUTO_START_INFERENCE and start_inference_loop():
                    print("✅ Auto inference started")
            else:
                print("❌ Model not loaded - auto inference not started")
        except Exception as e:
            print(f"❌ Model error: {e}")

    threading.Thread(target=_warm_model, daemon=True).start()

    yield

    print("🛑 Shutting down...")
    camera.stop()
    stop_inference_loop()


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Solar Cleaner Vision API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Firebase helper
# ──────────────────────────────────────────────────────────────────────────────

def push_to_firebase(payload: dict):
    try:
        url = f"{FIREBASE_DB_URL}/prediction.json"
        resp = requests.patch(url, json=payload, timeout=5)
        resp.raise_for_status()
        print(f"[Firebase] Pushed -> label={payload.get('label')}, score={payload.get('score')}")
    except Exception as err:
        print(f"[Firebase] Error pushing data: {err}")


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Prepare a BGR OpenCV frame for MobileNetV3Large.
    Steps:
      1. Resize to 224x224
      2. Convert BGR -> RGB
      3. Cast to float32
      4. Apply mobilenet_v3.preprocess_input (scales to [-1, 1])
      5. Add batch dimension -> shape (1, 224, 224, 3)
    """
    from tensorflow.keras.applications import mobilenet_v3
    resized = cv2.resize(frame, IMG_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = rgb.astype(np.float32)
    img = mobilenet_v3.preprocess_input(img)
    return np.expand_dims(img, axis=0)  # (1, 224, 224, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(frame: np.ndarray, allow_stub: bool = True) -> dict:
    """
    Run the solar panel cleanliness model on a single frame.
    Returns a dict with:
      - label            : "Clean" or "Dusty"
      - cls_probability  : float 0-1 (prob of being Clean)
      - score            : float 0-100 (100 = perfectly clean)
      - dust_severity    : float 1-10 (10 = very dusty)
      - stub             : bool — True if model not loaded (brightness-based estimate)
      - timestamp        : int ms
    """
    model = load_model()
    ts = int(time.time() * 1000)

    if model is None and not allow_stub:
        return {
            "label": "ModelUnavailable",
            "cls_probability": 0.0,
            "score": 0.0,
            "dust_severity": 0.0,
            "stub": False,
            "error": "Model not loaded",
            "timestamp": ts,
        }

    if model is None:
        # Stub mode: estimate cleanliness from frame brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))          # 0-255
        cleanliness = min(100.0, brightness / 2.55)  # 0-100
        label = "Clean" if cleanliness >= 50 else "Dusty"
        dust_severity = round(10 - (cleanliness / 10), 2)
        return {
            "label": label,
            "cls_probability": round(cleanliness / 100, 4),
            "score": round(cleanliness, 2),
            "dust_severity": dust_severity,
            "stub": True,
            "timestamp": ts,
        }

    # Real model inference
    try:
        inp = preprocess_frame(frame)
        predictions = model.predict(inp, verbose=0)

        # Accept dict or list outputs depending on model export format.
        if isinstance(predictions, dict):
            cls_prob = float(np.squeeze(predictions.get("cls", 0.0)))
            dust_val = float(np.squeeze(predictions.get("dust", 5.0)))
        elif isinstance(predictions, (list, tuple)):
            cls_prob = float(np.squeeze(predictions[0])) if len(predictions) >= 1 else 0.0
            dust_val = float(np.squeeze(predictions[1])) if len(predictions) >= 2 else 5.0
        else:
            arr = np.asarray(predictions)
            if arr.ndim >= 2 and arr.shape[-1] >= 2:
                cls_prob = float(arr[0][0])
                dust_val = float(arr[0][1])
            else:
                cls_prob = float(np.squeeze(arr))
                dust_val = 5.0

        label = "Clean" if cls_prob > 0.5 else "Dusty"
        score = round(cls_prob * 100, 2)        # 0-100
        dust_severity = round(float(np.clip(dust_val, 1, 10)), 2)

        return {
            "label": label,
            "cls_probability": round(cls_prob, 4),
            "score": score,
            "dust_severity": dust_severity,
            "stub": False,
            "timestamp": ts,
        }
    except Exception as e:
        print(f"[Inference] Error: {e}")
        return {
            "label": "Error",
            "cls_probability": 0.0,
            "score": 0.0,
            "dust_severity": 10.0,
            "stub": False,
            "error": str(e),
            "timestamp": ts,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Background inference loop
# ──────────────────────────────────────────────────────────────────────────────
_inference_running = False
_inference_thread: Optional[threading.Thread] = None


def _inference_loop():
    global _inference_running
    print(f"[Inference] Loop started - interval {INFERENCE_INTERVAL}s")
    while _inference_running:
        frame = camera.get_frame()
        if frame is not None:
            result = run_inference(frame, allow_stub=False)
            if "error" in result:
                print(f"[Inference] Skipped push: {result['error']}")
            else:
                push_to_firebase(result)
        time.sleep(INFERENCE_INTERVAL)
    print("[Inference] Loop stopped")


def stop_inference_loop():
    global _inference_running, _inference_thread
    _inference_running = False
    if _inference_thread and _inference_thread.is_alive():
        _inference_thread.join(timeout=2)


def start_inference_loop():
    global _inference_running, _inference_thread
    if _inference_running:
        return False
    _inference_running = True
    _inference_thread = threading.Thread(target=_inference_loop, daemon=True)
    _inference_thread.start()
    return True


# ──────────────────────────────────────────────────────────────────────────────
# MJPEG stream generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_mjpeg():
    """Yields MJPEG frames with overlay."""
    stream_interval = 1.0 / STREAM_TARGET_FPS
    while True:
        frame = camera.get_frame()
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "No camera signal",
                (180, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 200, 255),
                2,
            )

        # Overlay timestamp
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, ts, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(
            frame,
            "Solar Panel Feed",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            1,
        )

        ret, jpeg = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )
        time.sleep(stream_interval)


# ──────────────────────────────────────────────────────────────────────────────
# API Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    active_model_path = next((p for p in MODEL_CANDIDATE_PATHS if p.exists()), None)
    return {
        "status": "ok",
        "service": "solar-python-backend",
        "model_loaded": _model is not None,
        "model_initialized": _model_loaded,
        "camera_running": camera._running,
        "camera_index": camera._active_index,
        "camera_has_frame": camera.get_frame() is not None,
        "model_path": str(active_model_path) if active_model_path else None,
        "model_exists": active_model_path is not None,
        "model_error": _model_error,
    }


@app.get("/stream")
def video_stream():
    """MJPEG stream - embed as <img src='http://localhost:5001/stream'>"""
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive",
        },
    )


@app.post("/predict")
def predict():
    """Capture one frame, run inference, push result to Firebase."""
    frame = camera.get_frame()
    if frame is None:
        return {"success": False, "error": "No frame from camera"}
    result = run_inference(frame, allow_stub=False)
    if "error" in result:
        return {"success": False, **result}
    push_to_firebase(result)
    return {"success": True, **result}


@app.post("/start_inference")
def start_inference():
    if _inference_running:
        return {"status": "already_running"}
    if load_model() is None:
        return {"status": "error", "error": "Model not loaded"}
    start_inference_loop()
    return {"status": "started", "interval_seconds": INFERENCE_INTERVAL}


@app.post("/stop_inference")
def stop_inference():
    stop_inference_loop()
    return {"status": "stopped"}


@app.get("/snapshot")
def snapshot():
    """Return a single JPEG snapshot."""
    frame = camera.get_frame()
    if frame is None:
        return Response(content=b"", media_type="image/jpeg", status_code=503)
    _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=False)

