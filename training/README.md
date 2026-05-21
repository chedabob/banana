# Banana Detector — Training

Fine-tunes YOLOv11n on a banana ripeness dataset to produce a small ONNX model (~1.5 MB) that detects bananas and classifies their ripeness in one pass.

The trained model is loaded in the browser via `onnxruntime-web` — no TF.js needed, works on all platforms including iOS.

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

Training takes **1–2 hours** on M4. Progress prints to the terminal with mAP scores after each epoch.

When done, two files appear in `../models/`:
- `banana_yolo11n.onnx` — the quantised model (~1.5 MB)
- `metadata.json` — class names, accuracy scores, model name

## If mAP is below 0.85

Retrain with a slightly larger model — just one env var change, nothing else in the app changes:

```bash
MODEL=yolo11s.pt python train.py   # ~5.5MB INT8, noticeably better accuracy
MODEL=yolo11m.pt python train.py   # ~9.5MB INT8, best accuracy
```

The output ONNX and metadata.json are named automatically from the model (e.g. `banana_yolo11s.onnx`). Update the filename reference in `app.js` accordingly.

## Class mapping

The Roboflow dataset may use different label names than the app's internal STAGES ids (`unripe`, `nearly`, `perfect`, `ripe`, `overripe`).

`train.py` prints the raw dataset class names on first run. If the mapping looks wrong, update the `CLASS_MAP` dict at the top of `train.py` and re-run (the dataset is already downloaded, so only training reruns).

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

## Re-training from scratch

```bash
rm -rf runs/ dataset/
python train.py
```
