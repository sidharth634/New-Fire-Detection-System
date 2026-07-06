"""
Fire Detection Backend Server
Serves the HTML UI and processes webcam frames via YOLOv8 (best.pt)

Usage:
  pip install flask ultralytics opencv-python pillow
  python server.py

Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import os
import re
from datetime import datetime

app = Flask(__name__, static_folder='.')
CORS(app)  # Allow browser requests from same machine

# ── Load model ──────────────────────────────────────────────────────────────
# Search common locations for best.pt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_CANDIDATE_PATHS = [
    os.path.join(BASE_DIR, "best.pt"),
    os.path.join(BASE_DIR, "runs", "detect", "train6", "weights", "best.pt"),
    os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt"),
    os.path.join(BASE_DIR, "weights", "best.pt"),
]

MODEL_PATH = None
for _p in _CANDIDATE_PATHS:
    if os.path.exists(_p):
        MODEL_PATH = _p
        break

if MODEL_PATH is None:
    MODEL_PATH = "best.pt"  # fallback for error message

try:
    print("=" * 50)
    print("Current working directory:", os.getcwd())
    print("Trying to load:", os.path.abspath(MODEL_PATH))
    print("File exists:", os.path.exists(MODEL_PATH))

    model = YOLO(MODEL_PATH)

    print("[✓] Model loaded successfully!")
except Exception as e:
    print("[ERROR]", repr(e))
    model = None

# Detection sensitivity map
CONFIDENCE_MAP = {
    "balanced":  0.4,
    "high":      0.2,
    "strict":    0.6,
}

os.makedirs("alerts", exist_ok=True)

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'fire_monitoring_app.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')


@app.route('/detect', methods=['POST'])
def detect():
    """
    Accepts JSON: { "frame": "<base64 JPEG>", "mode": "balanced|high|strict" }
    Returns JSON: {
        "fire": bool,
        "detections": [{"label":"fire","conf":0.91,"box":[x1,y1,x2,y2]}, ...],
        "annotated": "<base64 JPEG with boxes drawn>"
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded", "fire": False, "detections": [], "annotated": None}), 503

    data = request.get_json(force=True)
    b64 = data.get("frame", "")
    mode = data.get("mode", "balanced").lower()

    # Strip data URI prefix if present
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Image decode failed: {e}"}), 400

    conf_thresh = CONFIDENCE_MAP.get(mode, 0.4)
    results = model.predict(frame, conf=conf_thresh, verbose=False)[0]

    detections = []
    annotated_frame = frame.copy()
    fire_detected = False

    if results.boxes is not None and len(results.boxes) > 0:
        for box, conf, cls in zip(
            results.boxes.xyxy.tolist(),
            results.boxes.conf.tolist(),
            results.boxes.cls.tolist()
        ):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)] if model.names else "fire"
            detections.append({"label": label, "conf": round(conf, 3), "box": [x1, y1, x2, y2]})

            # Draw box on annotated frame
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 60, 255), 3)
            tag = f"{label.upper()} {conf:.0%}"
            cv2.rectangle(annotated_frame, (x1, y1 - 26), (x1 + len(tag) * 10, y1), (0, 60, 255), -1)
            cv2.putText(annotated_frame, tag, (x1 + 4, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        fire_detected = True

        # Save snapshot
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"alerts/fire_{ts}.jpg", annotated_frame)

    # Encode annotated frame back to base64
    _, buf = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    return jsonify({
        "fire": fire_detected,
        "detections": detections,
        "annotated": annotated_b64
    })


@app.route('/status')
def status():
    return jsonify({
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "server": "Fire Monitoring Backend v1.0"
    })


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  🔥 Fire Monitoring System — Backend Server")
    print("="*55)
    print(f"  Model  : {MODEL_PATH} {'[LOADED]' if model else '[FAILED - check path]'}")
    print(f"  UI     : http://localhost:5000")
    print(f"  API    : http://localhost:5000/detect  (POST)")
    print("="*55 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
