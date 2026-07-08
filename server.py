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
import numpy as np
import cv2
import base64
import os
from datetime import datetime

app = Flask(__name__, static_folder=None)
CORS(app)  # Allow browser requests from any origin (e.g. your GitHub Pages site)

# ── Load model ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_CANDIDATE_PATHS = [
    os.path.join(BASE_DIR, "best.onnx"),
    os.path.join(BASE_DIR, "runs", "detect", "train6", "weights", "best.onnx"),
    os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.onnx"),
    os.path.join(BASE_DIR, "weights", "best.onnx"),
]

MODEL_PATH = None
for _p in _CANDIDATE_PATHS:
    if os.path.exists(_p):
        MODEL_PATH = _p
        break

if MODEL_PATH is None:
    MODEL_PATH = "best.onnx"

CLASS_NAMES = {0: "fire", 1: "smoke"}
net = None

try:
    print("=" * 50)
    print("Current working directory:", os.getcwd())
    print("Trying to load:", os.path.abspath(MODEL_PATH))
    print("File exists:", os.path.exists(MODEL_PATH))

    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Warm-up model
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(dummy, 1.0/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    net.forward()
    print("OpenCV DNN YOLOv8 warm-up complete")
    print("[OK] Model loaded successfully!")
except Exception as e:
    print("[ERROR]", repr(e))
    net = None

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
    return send_from_directory('.', 'index.html')


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
    if net is None:
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

    orig_h, orig_w = frame.shape[:2]
    conf_thresh = CONFIDENCE_MAP.get(mode, 0.4)

    # Preprocess image for YOLOv8 (640x640, BGR to RGB, scale by 1/255.0)
    blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)

    try:
        outputs = net.forward()
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {e}"}), 500

    # Parse detections (shape of output is (1, 6, 8400))
    output = np.squeeze(outputs[0])  # (6, 8400)
    output = output.T  # (8400, 6)

    boxes = []
    confidences = []
    class_ids = []

    # Map coordinates back to original frame size
    x_scale = orig_w / 640.0
    y_scale = orig_h / 640.0

    for row in output:
        scores = row[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence >= conf_thresh:
            x_center, y_center, w, h = row[0], row[1], row[2], row[3]
            
            # Convert center coords to top-left coords
            left = int(x_center - w / 2)
            top = int(y_center - h / 2)
            width = int(w)
            height = int(h)

            boxes.append([left, top, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    nms_thresh = 0.45
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

    detections = []
    annotated_frame = frame.copy()
    fire_detected = False

    if len(indices) > 0:
        flat_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
        for idx in flat_indices:
            box = boxes[idx]
            left, top, width, height = box
            conf = confidences[idx]
            class_id = class_ids[idx]
            label = CLASS_NAMES.get(class_id, "fire")

            # Scale to original coordinates
            x1 = int(left * x_scale)
            y1 = int(top * y_scale)
            x2 = int((left + width) * x_scale)
            y2 = int((top + height) * y_scale)

            # Clip coordinates
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))

            detections.append({
                "label": label,
                "conf": round(conf, 3),
                "box": [x1, y1, x2, y2]
            })

            # Draw box on annotated frame (matches original red-orange (0, 60, 255) color)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 60, 255), 3)
            tag = f"{label.upper()} {conf:.0%}"
            cv2.rectangle(annotated_frame, (x1, y1 - 26), (x1 + len(tag) * 10, y1), (0, 60, 255), -1)
            cv2.putText(annotated_frame, tag, (x1 + 4, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Set fire_detected to True if we have valid detections left after NMS
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
        "model_loaded": net is not None,
        "model_path": MODEL_PATH,
        "server": "Fire Monitoring Backend v1.0 (OpenCV DNN)"
    })


if __name__ == '__main__':
    print("\n" + "="*55)
    print("   Fire Monitoring System - Backend Server (OpenCV DNN)")
    print("="*55)
    print(f"  Model  : {MODEL_PATH} {'[LOADED]' if net else '[FAILED - check path]'}")
    print(f"  UI     : http://localhost:5000")
    print(f"  API    : http://localhost:5000/detect  (POST)")
    print("="*55 + "\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

