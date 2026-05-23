"""
Banana ripeness classifier — fine-tune YOLOv11n-cls on the Roboflow banana dataset.

The dataset is image-level classification (no bounding boxes), so this trains a
classifier. In the app, the existing COCO detector finds banana crops, and this
model classifies their ripeness — replacing the current HSL colour analysis.

Run from the training/ directory:
    python train.py

To retrain with a larger model if top-1 accuracy is insufficient:
    MODEL=yolo11s-cls.pt python train.py    # better accuracy, ~6MB INT8
    MODEL=yolo11m-cls.pt python train.py    # best accuracy, ~10MB INT8

Requires:
    pip install -r requirements.txt
    ROBOFLOW_API_KEY environment variable (free at roboflow.com)

Outputs:
    ../models/banana_yolo11n-cls.onnx  — INT8-quantised ONNX (~1MB)
    ../models/metadata.json            — class names, accuracy, training info
"""

import os, json, shutil, sys
from pathlib import Path

# ---- Config ----------------------------------------------------------------

ROBOFLOW_WORKSPACE = "roboflow-universe-projects"
ROBOFLOW_PROJECT   = "banana-ripeness-classification"
ROBOFLOW_VERSION   = 1

BASE_MODEL = os.environ.get("MODEL", "yolo11n-cls.pt")
EPOCHS     = 100
IMGSZ      = 224   # standard for classifiers; smaller = faster, sufficient for ripeness
BATCH      = 32
DEVICE     = "mps"  # 'mps' on Apple Silicon; 'cuda' on NVIDIA; 'cpu' fallback

DATASET_DIR = Path("./dataset")
OUTPUT_DIR  = Path("../models")

_model_stem = Path(BASE_MODEL).stem       # e.g. "yolo11n-cls"
ONNX_NAME   = f"banana_{_model_stem}.onnx"

# Map Roboflow folder names → our STAGES ids.
# The folder format uses directory names as class labels.
# Run once to see actual names, then update if needed.
CLASS_MAP = {
    "unripe"        : "unripe",
    "nearly-ripe"   : "nearly",
    "nearly_ripe"   : "nearly",
    "ripe"          : "perfect",
    "overripe"      : "ripe",
    "very-overripe" : "overripe",
    "very_overripe" : "overripe",
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
    dataset = project.version(ROBOFLOW_VERSION).download("folder", location=str(DATASET_DIR))
    print(f"Dataset saved to {DATASET_DIR}")
    return dataset


def remap_class_dirs(dataset_root: Path) -> list[str]:
    """
    Renames class subdirectories (train/valid/test/<class>) to match our STAGES ids.
    Safe to re-run: skips already-renamed dirs and cleans up __tmp_* leftovers from
    interrupted previous runs. Uses two-pass rename to avoid chain conflicts.
    """
    classes = set()
    for split in ("train", "valid", "test"):
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue

        # Clean up __tmp_* dirs left by interrupted previous runs
        for stale in split_dir.glob("__tmp_*__"):
            original_name = stale.name[6:-2]   # strip __tmp_ prefix and __ suffix
            mapped = CLASS_MAP.get(original_name.lower().replace(" ", "-"),
                                   CLASS_MAP.get(original_name.lower().replace(" ", "_"),
                                                 original_name.lower()))
            dst = split_dir / mapped
            if dst.exists():
                import shutil
                shutil.rmtree(stale)
                print(f"  removed stale {stale.name} (target {mapped} already exists)")
            else:
                stale.rename(dst)
                print(f"  recovered {stale.name} → {split}/{mapped}")

        renames = []
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            original = cls_dir.name
            mapped = CLASS_MAP.get(original.lower().replace(" ", "-"),
                                   CLASS_MAP.get(original.lower().replace(" ", "_"),
                                                 original.lower()))
            classes.add(mapped)
            if mapped != original:
                dst = split_dir / mapped
                if dst.exists():
                    print(f"  {split}/{original} → {split}/{mapped} (already done)")
                else:
                    renames.append((cls_dir, dst))
                    print(f"  {split}/{original} → {split}/{mapped}")

        # Pass 1: rename to unique temp names to avoid chain conflicts
        temp_map = []
        for src, dst in renames:
            tmp = src.parent / f"__tmp_{src.name}__"
            src.rename(tmp)
            temp_map.append((tmp, dst))

        # Pass 2: rename from temp to final names
        for tmp, dst in temp_map:
            tmp.rename(dst)

    class_names = sorted(classes)
    print(f"Classes: {class_names}")
    return class_names


def train(dataset_root: Path):
    from ultralytics import YOLO
    print(f"\nFine-tuning {BASE_MODEL} for {EPOCHS} epochs on {DEVICE}…")
    model = YOLO(BASE_MODEL)
    results = model.train(
        data=str(dataset_root),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project="runs",
        name="banana",
        exist_ok=True,
        # Augmentation — aggressive colour/geometry variation to handle real-world
        # lighting (supermarket LED, shadows) and crops of bunches vs single bananas
        hsv_h=0.05,    # ±5% hue shift handles CRI differences between light sources
        hsv_s=0.7,     # wide saturation range covers washed-out vs vivid conditions
        hsv_v=0.5,     # brightness range handles shadows and harsh store lighting
        flipud=0.1,
        fliplr=0.5,
        degrees=15,    # banana bunches appear at many angles on a shelf
        translate=0.1,
        scale=0.3,     # crops from COCO-SSD vary in how much banana fills the frame
    )
    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = save_dir / "weights" / "last.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")
    return best_weights


def validate(weights: Path, dataset_root: Path):
    from ultralytics import YOLO
    print("\nValidating on test set…")
    model = YOLO(str(weights))
    metrics = model.val(data=str(dataset_root), split="test", imgsz=IMGSZ, device=DEVICE)
    top1 = float(metrics.top1)
    top5 = float(metrics.top5)
    print(f"Top-1 accuracy: {top1:.3f}   Top-5 accuracy: {top5:.3f}")
    if top1 < 0.80:
        print(f"⚠  Top-1 below 0.80 — consider a larger model:")
        print(f"   MODEL=yolo11s-cls.pt python train.py")
    return top1, top5


def export_onnx(weights: Path, class_names: list, top1: float, top5: float):
    from ultralytics import YOLO
    from onnxruntime.quantization import quantize_dynamic, QuantType
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: export FP32 ONNX from Ultralytics
    print(f"\nExporting ONNX (FP32)…")
    model = YOLO(str(weights))
    fp32_path = Path(model.export(format="onnx", imgsz=IMGSZ, simplify=True))

    # Step 2: dynamic INT8 quantisation via onnxruntime
    dest = OUTPUT_DIR / ONNX_NAME
    print(f"Quantising to INT8 → {dest}…")
    quantize_dynamic(str(fp32_path), str(dest), weight_type=QuantType.QInt8)
    fp32_path.unlink()  # remove intermediate FP32

    size_mb = dest.stat().st_size / 1e6
    print(f"Saved {dest}  ({size_mb:.2f} MB)")

    meta = {
        "model"      : _model_stem,
        "type"       : "classification",
        "onnx_file"  : ONNX_NAME,
        "classes"    : class_names,
        "imgsz"      : IMGSZ,
        "top1"       : round(top1, 4),
        "top5"       : round(top5, 4),
        "trained_on" : f"Roboflow {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT} v{ROBOFLOW_VERSION}",
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata written to {OUTPUT_DIR / 'metadata.json'}")

    return dest, size_mb


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=None,
                        help="Skip training and export these existing weights directly")
    args = parser.parse_args()

    # Ensure dataset is present (needed for class names and validation)
    if not (DATASET_DIR / "train").exists():
        download_dataset()
    else:
        print(f"Dataset already present at {DATASET_DIR}, skipping download.")

    class_names = remap_class_dirs(DATASET_DIR)

    if args.weights:
        best_weights = args.weights.resolve()
        if not best_weights.exists():
            sys.exit(f"Weights not found: {best_weights}\n"
                     f"(script runs from {Path.cwd()}, path resolved from there)")
        print(f"\nSkipping training — using existing weights: {best_weights}")
    else:
        best_weights = train(DATASET_DIR)

    top1, top5 = validate(best_weights, DATASET_DIR)
    dest, size_mb = export_onnx(best_weights, class_names, top1, top5)

    print(f"\n✓  Done!  {dest}  ({size_mb:.2f} MB)")
    print(f"   Top-1 accuracy = {top1:.3f}")
    if top1 < 0.85:
        print(f"\nTip: accuracy below 0.85. Retry with a larger model:")
        print(f"   MODEL=yolo11s-cls.pt python train.py")
    print(f"\nNext: commit models/ then wire up app.js classification integration.")
