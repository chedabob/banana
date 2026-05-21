# Banana Detector — Training

Fine-tunes YOLOv8n on a banana ripeness dataset to produce a small ONNX model (~1.5 MB) that detects bananas and classifies their ripeness in one pass.

## Requirements

- Mac with Apple Silicon (M1/M2/M3/M4) — uses `mps` device by default
- Python 3.10+
- Free [Roboflow account](https://roboflow.com) for dataset download

## Setup

```bash
cd training/
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
export ROBOFLOW_API_KEY=your_key_here
python train.py
```

Training takes **1–3 hours** on M4. Progress prints to the terminal with mAP scores after each epoch.

When done, two files appear in `../models/`:
- `banana_yolov8n.onnx` — the quantised model (~1.5 MB)
- `metadata.json` — class names and accuracy scores

## Class mapping

The Roboflow dataset may use different label names than the app's internal STAGES ids (`unripe`, `nearly`, `perfect`, `ripe`, `overripe`).

`train.py` prints the raw dataset class names on first run. If the mapping looks wrong, update the `CLASS_MAP` dict at the top of `train.py` and re-run (training output is cached, so only the export step re-runs if weights already exist).

## Checking results

After training you can inspect predictions:

```bash
python -c "
from ultralytics import YOLO
m = YOLO('runs/banana/weights/best.pt')
m.predict('path/to/banana.jpg', save=True)
"
```

Results are saved to `runs/banana/predict/`.

## Re-training

To start fresh:
```bash
rm -rf runs/ dataset/
python train.py
```
