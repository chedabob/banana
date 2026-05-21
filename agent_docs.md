# Banana Detector — Agent Session Notes

## What This App Does

Progressive Web App at `chedabob.github.io/banana`. Points camera at bananas and classifies ripeness into 5 stages: unripe / nearly / perfect / ripe / overripe. Designed primarily for scanning **many bunches on a supermarket shelf** (not just one banana at a time).

Deployed branch: `main`. Dev branch specified in session: `claude/banana-detector-pwa-d8LOo` (but all work has been on `main`).

---

## Architecture

### Two-stage pipeline
1. **Detector** (bounding boxes): COCO-SSD on iOS, YOLOS-tiny on desktop/Android
2. **Classifier** (ripeness): Custom YOLOv11s-cls trained on Roboflow banana dataset, served as ONNX

### Key files
| File | Role |
|------|------|
| `app.js` | All app logic — model loading, camera, detection, classification, UI |
| `index.html` | App shell |
| `style.css` | Styling — viewfinder is `max-height:70vh` |
| `sw.js` | Service worker — cache `banana-detector-v6` |
| `models/banana_yolo11s-cls.onnx` | ~6MB INT8-quantised classifier |
| `models/metadata.json` | Class names, accuracy, model info |
| `training/train.py` | Full training pipeline |
| `mise.toml` | Python env management — `mise run train` |

### ONNX inference
```javascript
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/+esm';
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/';
ort.env.wasm.numThreads = 1;
```
Input tensor: `{ images: Float32Array [1,3,224,224] }`, normalised to [0,1].  
Output tensor: `output0` (or first key — use fallback: `outputs.output0 ?? outputs[Object.keys(outputs)[0]]`).

### Tiling (shelf scanning)
For images larger than 640×640, the detector runs on 640×640 tiles with 320px stride (50% overlap). For a 1280×960 camera frame this gives 6 tiles. NMS (IoU 0.35) merges boxes across tiles.

```javascript
const TILE_SIZE = 640;
const TILE_STRIDE = 320;
```

### Loading order
Detector and classifier load **in parallel** (detector is fire-and-forget, does not block splash). Splash hides when classifier finishes loading (or after 6s on error). If classifier fails, app continues with detector + HSL colour fallback.

---

## Model: YOLOv11s-cls

- Dataset: `roboflow-universe-projects/banana-ripeness-classification` v1 — 5,616 images, 5 classes
- Base model: `yolo11s-cls.pt` (not `n` — `n` got <80% accuracy)
- Accuracy: **99.1% Top-1**
- Export: FP32 via Ultralytics → INT8 via `onnxruntime.quantization.quantize_dynamic`
- File: `models/banana_yolo11s-cls.onnx` (~6MB)
- Classes (alphabetical, as stored in metadata.json): `["nearly", "overripe", "perfect", "ripe", "unripe"]`
- Note: classifier outputs are class-indexed by this alphabetical order, not STAGES order

### Class → STAGES mapping
```javascript
const STAGES = [
    { id:'unripe', ... },
    { id:'nearly', ... },
    { id:'perfect', ... },
    { id:'ripe', ... },
    { id:'overripe', ... }
];
// lookup: STAGES.find(s => s.id === classifierClasses[idx])
```

---

## Training Pipeline (Mac Mini M4)

### Quick start
```bash
export ROBOFLOW_API_KEY=your_key
mise run train                  # or: cd training && .venv/bin/python train.py
```

### Mise tasks
```toml
[tasks.install]  # creates training/.venv + installs requirements.txt
[tasks.train]    # depends=install, runs train.py from training/ dir
```

### Skipping retraining
```bash
cd training && .venv/bin/python train.py --weights ../runs/classify/runs/banana/weights/best.pt
```
Weights live at `training/runs/classify/runs/banana/weights/` (Ultralytics puts classify runs under `runs/classify/`).

### Key train.py details
- `CLASS_MAP` renames Roboflow folder names to our `STAGES` ids (two-pass rename to avoid chain conflicts)
- `remap_class_dirs` is idempotent — safe to re-run if interrupted
- `export_onnx`: exports FP32 ONNX first, then quantises with `onnxruntime.quantization.quantize_dynamic`
- Do NOT use `int8=True` in `model.export()` — Ultralytics doesn't support that for ONNX

---

## Known Issues / Open Questions

### Classification accuracy
On a banana ripeness chart (#1 very green → #7 brown), model called most "Perfect" except #2. Two possible causes:
1. **Output tensor name bug** (now fixed): `quantize_dynamic` might rename the output tensor; fixed with `outputs.output0 ?? outputs[Object.keys(outputs)[0]]`
2. **Distribution shift**: model trained on full bunch images, tested on single-banana crops — may affect accuracy. Not yet fully diagnosed.

Recommendation: test classification on sample images after the tensor name fix, look at browser console for any classifier errors.

### Intermittent model load failures
User reports "mode load failed" on refresh. Possible causes:
- WASM memory pressure on iOS after multiple refreshes
- CDN timing (ort WASM files load from jsdelivr)
- Not yet reliably reproduced to diagnose root cause

### Shelf detection
Tiling is implemented and should work. Key thing to verify: does the detector actually find all bunches across a full shelf? COCO-SSD threshold is 0.4 — may need lowering if far-away bunches are missed. YOLOS-tiny is better at small objects.

---

## Decisions Already Made

- **No AI-trains-AI / distillation for v1**: Train directly on human-annotated Roboflow data. No bias propagation from VLMs or larger teacher models.
- **No detection-only model**: Using two-stage (COCO/YOLOS for boxes + custom classifier for ripeness). A single YOLOv11n detection model would need bounding-box annotations which the Roboflow dataset doesn't have.
- **Single ONNX path, not TF.js for iOS**: onnxruntime-web@1.26.0 works on iOS WASM; previous iOS issues were with 200MB+ models, not ~6MB ONNX.
- **numThreads = 1**: Avoids COOP/COEP header requirement for SharedArrayBuffer.
- **updateViaCache: 'none'**: Service worker re-validates app files on every load, so cached stale JS can't get stuck.

---

## Service Worker

Cache name: `banana-detector-v6`. Bump to `v7` if you need to force-clear on all clients.  
ONNX is precached via `Promise.allSettled` (non-fatal — SW installs even if ONNX fetch is slow).  
`updateViaCache: 'none'` in SW registration ensures app.js is always re-validated.

---

## Potential Next Steps

1. **Verify classification fix** (output tensor name) — load app, scan banana, check console for classifier errors
2. **Lower COCO-SSD threshold** from 0.4 → 0.3 if far-away shelf bananas are missed
3. **Add console debug for tile count** to verify tiling fires on large images
4. **Retrain with larger model** if classification accuracy is still poor after tensor fix — try `yolo11m-cls.pt`
5. **Detection-only classifier** (longer term) — train YOLOv11n detection model if bounding-box annotations can be sourced
