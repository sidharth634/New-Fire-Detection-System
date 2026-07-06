# 🔥 Fire Monitoring System — Setup Guide

## What this is
A fully integrated fire monitoring dashboard that:
- Uses your existing `fire_monitoring_ui.html` design (sidebar, grid, modals)
- Streams your **laptop webcam live into Camera 1 / Floor 1**
- Sends frames to a Python backend running **YOLOv8 (best.pt)**
- Draws **red detection boxes** around fire/smoke in real-time
- Triggers the **emergency modal, voice alert, and event log** on detection

---

## File Structure
Place all files in the SAME folder:

```
your-project/
├── fire_monitoring_app.html   ← Main app (open this in browser)
├── server.py                  ← Python backend (run this first)
├── styles.css                 ← Your original CSS (unchanged)
├── styles.scss                ← Your original SCSS (optional)
└── runs/
    └── detect/
        └── train6/
            └── weights/
                └── best.pt    ← Your trained YOLOv8 model
```

---

## Setup & Run

### Step 1 — Install dependencies
```bash
pip install flask flask-cors ultralytics opencv-python pillow
```

### Step 2 — Start the backend server
```bash
python server.py
```
You should see:
```
  🔥 Fire Monitoring System — Backend Server
  Model  : runs/detect/train6/weights/best.pt
  UI     : http://localhost:5000
```

### Step 3 — Open the app
Open your browser and go to:
```
http://localhost:5000
```
Or just double-click `fire_monitoring_app.html` (backend must still be running).

---

## How to Use

1. **Start webcam** — Click "▶ Start Webcam" button on the Dashboard
2. **Allow camera** — Browser will ask for camera permission, click Allow
3. **Watch Camera 1** — Floor 1 tile shows your live webcam feed
4. **Detection runs automatically** — Every 500ms a frame is sent to YOLOv8
5. **Fire detected?** — Red box appears, modal pops up, voice alert plays (if enabled)
6. **Click any camera tile** → opens full-screen view in Cameras tab

---

## Controls

| Control | Description |
|---|---|
| Detection Mode | Balanced (0.4 conf) / High Sensitivity (0.2) / Strict (0.6) |
| Voice Alert toggle | Enables browser speech synthesis on fire detection |
| Start/Stop Webcam | Starts or stops the live webcam + detection loop |
| Test Alert button | Manually triggers the emergency modal |

---

## API Endpoint (for integration)
```
POST http://localhost:5000/detect
Body: { "frame": "<base64 JPEG>", "mode": "balanced|high|strict" }
Response: { "fire": bool, "detections": [...], "annotated": "<base64 JPEG>" }
```

---

## Notes
- Detection runs at ~2 fps to keep CPU usage manageable
- Detected frames are auto-saved to `alerts/` folder
- Camera 2–8 show hallway simulations (extend with more webcams/RTSP streams)
