"""
Banana ripeness detector — fine-tune YOLOv8n on the Roboflow banana dataset.

Run from the training/ directory:
    python train.py

Requires:
    pip install -r requirements.txt
    ROBOFLOW_API_KEY environment variable (free at roboflow.com)

Outputs:
    ../models/banana_yolov8n.onnx     — INT8-quantised ONNX for all platforms (~1.5MB)
    ../models/metadata.json           — class names, mAP, training info

The ONNX model is loaded in the app via onnxruntime-web.
"""

import os, json, shutil, sys
from pathlib import Path

# ---- Config ----------------------------------------------------------------

ROBOFLOW_WORKSPACE = "roboflow-universe-projects"
ROBOFLOW_PROJECT   = "banana-ripeness-classification"
ROBOFLOW_VERSION   = 1          # bump if the dataset has been updated

EPOCHS   = 100
IMGSZ    = 640
BATCH    = 16                   # reduce to 8 if you hit memory issues
DEVICE   = "mps"                # 'mps' on Apple Silicon; 'cuda' on NVIDIA; 'cpu' fallback

DATASET_DIR = Path("./dataset")
OUTPUT_DIR  = Path("../models")

# Map Roboflow class names → our STAGES ids.
# Preview the dataset first (roboflow.com) to confirm actual label names,
# then update this dict if they differ.
CLASS_MAP = {
    # Roboflow label   : STAGES id
    "unripe"           : "unripe",
    "nearly-ripe"      : "nearly",
    "nearly_ripe"      : "nearly",
    "ripe"             : "perfect",    # Roboflow "ripe" = our "perfect" sweet spot
    "overripe"         : "ripe",       # Roboflow "overripe" = our "ripe" stage
    "very-overripe"    : "overripe",
    "very_overripe"    : "overripe",
    # Add more mappings here if the dataset uses different labels
}

# ---------------------------------------------------------------------------

def download_dataset():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("Set ROBOFLOW_API_KEY env var. Get one free at roboflow.com.")

    print("Downloading dataset from Roboflow…")
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download("yolov8", location=str(DATASET_DIR))
    print(f"Dataset saved to {DATASET_DIR}")
    return dataset


def patch_class_names(data_yaml: Path):
    """
    Rewrites the dataset's data.yaml so class names match our STAGES ids.
    Prints the original names so you can update CLASS_MAP if needed.
    """
    import yaml
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    original = cfg.get("names", [])
    print(f"Dataset class names: {original}")

    mapped = []
    for name in original:
        canonical = CLASS_MAP.get(name.lower().replace(" ", "-"),
                                  CLASS_MAP.get(name.lower().replace(" ", "_"), name.lower()))
        mapped.append(canonical)

    if mapped != original:
        cfg["names"] = mapped
        with open(data_yaml, "w") as f:
            yaml.dump(cfg, f)
        print(f"Remapped → {mapped}")
    else:
        print("Class names look good, no remapping needed.")

    return mapped


def train(data_yaml: Path):
    from ultralytics import YOLO
    print(f"\nFine-tuning YOLOv8n for {EPOCHS} epochs on {DEVICE}…")
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project="runs",
        name="banana",
        exist_ok=True,
        # Augmentation — helps generalise to store shelves, varied lighting
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        flipud=0.1,
        fliplr=0.5,
        mosaic=0.8,
    )
    best_weights = Path("runs/banana/weights/best.pt")
    print(f"\nTraining complete. Best weights: {best_weights}")
    return best_weights, results


def validate(weights: Path, data_yaml: Path):
    from ultralytics import YOLO
    print("\nValidating on test set…")
    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split="test", imgsz=IMGSZ, device=DEVICE)
    map50    = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    print(f"mAP@50: {map50:.3f}   mAP@50-95: {map50_95:.3f}")
    if map50 < 0.75:
        print("⚠  mAP@50 below 0.75 — consider more epochs or checking class mapping.")
    return map50, map50_95


def export_onnx(weights: Path, class_names: list, map50: float, map50_95: float):
    from ultralytics import YOLO
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUTPUT_DIR / "banana_yolov8n.onnx"

    print(f"\nExporting ONNX (INT8 quantised)…")
    model = YOLO(str(weights))
    # half=False because int8=True is the quantisation path; imgsz must match training
    export_path = model.export(format="onnx", imgsz=IMGSZ, int8=True, simplify=True)
    shutil.copy(export_path, dest)
    size_mb = dest.stat().st_size / 1e6
    print(f"Saved {dest}  ({size_mb:.2f} MB)")

    # Write metadata for the app
    meta = {
        "classes"    : class_names,
        "imgsz"      : IMGSZ,
        "map50"      : round(map50, 4),
        "map50_95"   : round(map50_95, 4),
        "trained_on" : f"Roboflow {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT} v{ROBOFLOW_VERSION}",
    }
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata written to {meta_path}")

    return dest, size_mb


if __name__ == "__main__":
    # 1. Download dataset
    if not (DATASET_DIR / "data.yaml").exists():
        download_dataset()
    else:
        print(f"Dataset already present at {DATASET_DIR}, skipping download.")

    data_yaml = DATASET_DIR / "data.yaml"

    # 2. Patch class names to match our STAGES ids
    class_names = patch_class_names(data_yaml)

    # 3. Train
    best_weights, _ = train(data_yaml)

    # 4. Validate on test set
    map50, map50_95 = validate(best_weights, data_yaml)

    # 5. Export ONNX
    dest, size_mb = export_onnx(best_weights, class_names, map50, map50_95)

    print(f"\n✓  Done!  {dest}  ({size_mb:.2f} MB)")
    print(f"   mAP@50 = {map50:.3f}   mAP@50-95 = {map50_95:.3f}")
    print(f"\nNext: commit models/ and update WORKER_URL in app.js if using Cloudflare.")
