# Banana Detector — Session Notes

## What This App Does

Progressive Web App at `chedabob.github.io/banana`. Points the device camera at bananas and classifies ripeness into 5 stages: **unripe / nearly / perfect / ripe / overripe**. Designed primarily for scanning **many bunches on a supermarket shelf** — not just one banana at a time.

Deployed branch: `main`.

---

## Architecture

### Two-stage pipeline
1. **Detector** (bounding boxes): COCO-SSD on iOS, YOLOS-tiny on desktop/Android — finds banana crops
2. **Classifier** (ripeness): Custom YOLOv11s-cls trained on Roboflow banana dataset, served as ONNX — classifies each crop

Both load in parallel on boot. Detector is fire-and-forget; classifier controls the splash screen. If the classifier fails, the app continues with detector + HSL colour fallback.

### Ripeness stages
```javascript
const STAGES = [
    { id:'unripe',   emoji:'🟢', pos: 7  },
    { id:'nearly',   emoji:'🟡', pos: 28 },
    { id:'perfect',  emoji:'⭐',  pos: 50 },
    { id:'ripe',     emoji:'✅',  pos: 72 },
    { id:'overripe', emoji:'🍞', pos: 92 }
];
```

### Tiling (shelf scanning)
For images wider/taller than 640px the detector runs on 640×640 tiles with 320px stride (50% overlap). A 1280×960 camera frame → 6 tiles. NMS (IoU 0.35) merges boxes across tiles.

```javascript
const TILE_SIZE = 640;
const TILE_STRIDE = 320;
```

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

---

## ONNX Inference

```javascript
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/+esm';
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/';
ort.env.wasm.numThreads = 1;  // avoids COOP/COEP SharedArrayBuffer requirement
```

Input tensor: `{ images: Float32Array [1,3,224,224] }`, normalised to [0,1], CHW layout.

Output tensor lookup — robust to `quantize_dynamic` renaming the output:
```javascript
const outputs = await classifierSession.run({ images: tensor });
const outputTensor = outputs.output0 ?? outputs[Object.keys(outputs)[0]];
const logits = outputTensor.data;
```
This was a bug fix: if `output0` didn't exist the catch silently swallowed it, returning `null`, and HSL fallback then classified every yellow banana as "Perfect".

---

## Model: YOLOv11s-cls

- Dataset: `roboflow-universe-projects/banana-ripeness-classification` v1 — 5,616 images, 5 classes, human-annotated, pre-split (~70/20/10%)
- Base model: `yolo11s-cls.pt` — `n` was tried first but gave <80% Top-1 accuracy
- Accuracy: **99.1% Top-1**
- Export: FP32 via Ultralytics → INT8 via `onnxruntime.quantization.quantize_dynamic`
- File: `models/banana_yolo11s-cls.onnx` (~6MB)
- Classes stored alphabetically in `metadata.json`: `["nearly", "overripe", "perfect", "ripe", "unripe"]`

Class lookup: `STAGES.find(s => s.id === classifierClasses[idx])` — maps classifier output index to STAGES entry by `id` string, not position.

### Why not AI-trains-AI / distillation
Explicitly decided against for v1. Training on human-annotated Roboflow data only — no VLM auto-labelling, no knowledge distillation from a larger teacher model. Avoids bias transfer from AI-generated labels becoming ground truth. Can revisit if direct training accuracy plateaus.

---

## Training Pipeline (Mac Mini M4)

### Quick start
```bash
export ROBOFLOW_API_KEY=your_key
mise run train
# or directly:
cd training && .venv/bin/python train.py
```

### Mise tasks
```toml
[tasks.install]  # creates training/.venv + installs requirements.txt
[tasks.train]    # depends=install, runs train.py from training/ dir
```
The `install` task creates the venv explicitly via `python -m venv training/.venv` — `{root}` doesn't expand in Mise `run` strings, so all paths must be relative or absolute.

### Skipping retraining
```bash
cd training && .venv/bin/python train.py --weights ../runs/classify/runs/banana/weights/best.pt
```
Ultralytics puts classification runs under `runs/classify/` (not `runs/`), so weights are at `training/runs/classify/runs/banana/weights/`.

### Key train.py details
- Download format: `folder` (not `yolov8` — that's only for detection projects)
- `CLASS_MAP` renames Roboflow folder names to our STAGES ids
- `remap_class_dirs` is idempotent — two-pass rename with `__tmp_` prefix avoids chain conflicts; safe to re-run after interruption
- Export: FP32 ONNX first via Ultralytics, then `onnxruntime.quantization.quantize_dynamic` for INT8. Do NOT use `int8=True` in `model.export()` — Ultralytics doesn't support it for ONNX format.
- `results.save_dir` gives the actual weights directory — don't hardcode the path

---

## Service Worker

Cache name: `banana-detector-v6`. Bump to `v7` to force-clear on all clients.

- ONNX precached via `Promise.allSettled` (non-fatal — SW installs even if ONNX fetch times out)
- `updateViaCache: 'none'` in SW registration so app.js is always re-validated on load
- iOS cache clearing: Settings → Safari → Advanced → Website Data → delete site entry (users can't clear from within the browser)

---

## Lighting & Torch Toggle (discussed, decided against)

The user noted that bananas look yellower (riper) under supermarket LED/fluorescent lighting (poor CRI) but greener/unriper in natural light. They asked whether a torch/flash button would help with consistent classification.

**Decision: deferred.** Reasons:
- The classifier uses learned texture/shape/colour patterns, not raw HSL. At 99.1% Top-1 it's already robust to lighting variation.
- Torch creates its own colour cast (warm white, harsh shadows, specular highlights on shiny banana skin) — also not present in training data, so it substitutes one colour bias for another, not a net improvement.
- For the primary use case (supermarket shelf scanning), torch range is ~0.5–1m — useless at shelf distance. The viewfinder would illuminate only the nearest bunches.

**What torch IS good for:** close-up single-banana scanning in a dim kitchen. Worth adding later as a convenience feature.

**If adding later — implementation notes:**
- API: `track.applyConstraints({ advanced: [{ torch: true }] })`
- Capability check: `track.getCapabilities?.().torch` — returns `true` on Chrome Android and Safari iOS 16.4+; undefined on desktop
- Add `btn-torch` button, hidden until camera starts and capability confirmed
- Toggle state in a `let torchOn = false` variable
- Reset state in `stopCamera()` — `track.stop()` kills the hardware torch automatically, just need to reset the variable and button UI
- Hide button on capture, restore on retake

---

## Known Issues

### Classification accuracy
On a banana ripeness chart (#1 very green → #7 brown), the model called most bananas "Perfect" except #2. Two possible causes investigated:

1. **Output tensor name bug (now fixed):** `quantize_dynamic` may rename `output0`; silent catch returned `null` → HSL fallback → yellow bananas all classified as "Perfect". Fixed with robust tensor lookup.
2. **Distribution shift:** model trained on full bunch images, tested on individual banana crops. May still affect edge cases — not fully diagnosed. Test after the tensor fix to see if accuracy improves.

### Intermittent model load failures
User reports "mode load failed" appearing on refresh occasionally. Possible causes:
- WASM memory pressure on iOS after multiple rapid refreshes
- CDN timing for onnxruntime WASM files from jsdelivr
- Not reliably reproduced. Worth investigating if it persists after the parallel-loading fix.

### Shelf detection coverage
Tiling is implemented but was not visibly working when the classifier failed (because the old code only started the detector after the classifier succeeded). Fixed in the parallel-loading commit. Separate concern: COCO-SSD detection threshold is 0.4 — may miss far-away or partially obscured bunches.

---

## Decisions Log

| Decision | Outcome | Rationale |
|----------|---------|-----------|
| AI-trains-AI / distillation | Declined for v1 | Bias propagation risk; direct training got 99.1% anyway |
| YOLOv11n vs YOLOv11s | Use `s` | `n` gave <80% Top-1; `s` got 99.1% |
| Single ONNX vs TF.js for iOS | Single ONNX | ort-web@1.26.0 works on iOS WASM; TF.js was only needed when model was 200MB+ |
| Torch toggle | Deferred | Doesn't help shelf scanning; training data mismatch |
| `numThreads = 1` | Keep | Avoids COOP/COEP headers for SharedArrayBuffer |
| Detector loads in parallel | Done | Was previously blocked on classifier — caused tiling to never run on failure |

---

## Potential Next Steps

1. **Verify classification fix** — test with a very green banana and check console for any classifier errors
2. **Lower COCO-SSD threshold** from 0.4 → 0.3 if far-away shelf bunches are missed
3. **Add console debug for tile count** to confirm tiling fires on large images (`console.log(`tiling: ${xs.length}×${ys.length} tiles`)`)
4. **Retrain with larger model** if accuracy is still poor after tensor fix — `MODEL=yolo11m-cls.pt python train.py`
5. **Torch toggle button** — see notes above; straightforward ~30 min addition
6. **Detection model** (longer term) — a single YOLOv11n detection model with bounding-box + ripeness class in one pass would be cleaner but needs bounding-box annotations (the Roboflow dataset is classification-only)
