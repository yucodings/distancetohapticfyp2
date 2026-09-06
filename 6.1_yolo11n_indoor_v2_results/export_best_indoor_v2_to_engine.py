#!/usr/bin/env python3
"""Export the 6.1 YOLO11n checkpoint to a Jetson TensorRT engine.

Run this program on the same Orin Nano that will execute the engine. The
result is a static batch-1, 640x640, FP16 detection plan with raw YOLO output;
NMS remains in the 7.0 application postprocessor.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

# Training/export does not require a desktop plotting window.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import tensorrt as trt
import torch
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "best_indoor_v2.pt"
EXPECTED_IMAGE_SIZE = 640
EXPECTED_CLASSES = (
    "chair",
    "cleaning_cart",
    "handrail",
    "person",
    "pillars",
    "ramp",
    "squat-toilet",
    "stairs",
    "urinals",
    "vending_machine",
    "wet_floor_sign",
    "bicycle",
    "bench",
    "cat",
    "dog",
    "backpack",
    "umbrella",
    "handbag",
    "suitcase",
    "bottle",
    "bed",
    "dining table",
    "couch",
    "toilet",
    "potted plant",
    "sink",
    "refrigerator",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"checkpoint to export (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=1.0,
        help="TensorRT builder workspace limit in GiB (default: 1.0)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing .engine with the same name",
    )
    parser.add_argument(
        "--skip-inference-check",
        action="store_true",
        help="skip one blank-frame inference after structural verification",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def available_memory_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        pass
    return None


def ordered_names(raw_names) -> tuple[str, ...]:
    if isinstance(raw_names, dict):
        return tuple(
            str(raw_names[index])
            for index in sorted(raw_names, key=lambda value: int(value))
        )
    return tuple(str(name) for name in raw_names)


def validate_engine_structure(engine_path: Path) -> None:
    """Validate the Ultralytics prefix and TensorRT input/output contract."""
    with engine_path.open("rb") as stream:
        length_bytes = stream.read(4)
        if len(length_bytes) != 4:
            fail("engine metadata prefix is truncated")
        metadata_length = int.from_bytes(length_bytes, "little", signed=False)
        if not 2 <= metadata_length <= 1_000_000:
            fail(f"invalid engine metadata length: {metadata_length}")
        try:
            metadata = json.loads(stream.read(metadata_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            fail(f"invalid engine metadata: {error}")
        plan = stream.read()

    names = ordered_names(metadata.get("names") or {})
    args = metadata.get("args") or {}
    if metadata.get("task") != "detect":
        fail(f"engine task is not detection: {metadata.get('task')!r}")
    if int(metadata.get("batch", 0)) != 1:
        fail(f"engine batch is not 1: {metadata.get('batch')!r}")
    if tuple(metadata.get("imgsz") or ()) != (EXPECTED_IMAGE_SIZE,) * 2:
        fail(f"engine image size is not 640x640: {metadata.get('imgsz')!r}")
    if not bool(args.get("half")):
        fail("engine metadata does not report FP16 optimization")
    if bool(args.get("nms")):
        fail("engine has embedded NMS; 7.0 expects raw YOLO output")
    if names != EXPECTED_CLASSES:
        fail("engine class names/order do not match the 6.1 deployment contract")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        fail("TensorRT could not deserialize the exported plan")
    tensors = {
        engine.get_tensor_name(index): tuple(
            engine.get_tensor_shape(engine.get_tensor_name(index))
        )
        for index in range(engine.num_io_tensors)
    }
    if tensors.get("images") != (1, 3, EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE):
        fail(f"unexpected TensorRT input shape: {tensors}")
    if tensors.get("output0") != (1, 4 + len(EXPECTED_CLASSES), 8400):
        fail(f"unexpected TensorRT output shape: {tensors}")
    print("Structural verification passed:")
    print(f"  metadata: batch 1, FP16, NMS off, {len(names)} classes")
    print(f"  tensors : {tensors}")


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    engine_path = model_path.with_suffix(".engine")
    if not model_path.is_file():
        fail(f"checkpoint not found: {model_path}")
    if model_path.suffix.lower() != ".pt":
        fail(f"expected a .pt checkpoint: {model_path}")
    if args.workspace <= 0:
        fail("--workspace must be positive")
    if engine_path.exists() and not args.force:
        fail(f"engine already exists: {engine_path} (use --force to replace it)")
    if platform.machine() != "aarch64":
        fail("run this exporter on the target Jetson/Orin Nano (aarch64)")
    if not torch.cuda.is_available():
        fail("PyTorch cannot access the Jetson CUDA GPU")

    model = YOLO(str(model_path), task="detect")
    names = ordered_names(model.names)
    if names != EXPECTED_CLASSES:
        fail(
            f"checkpoint has the wrong class contract: expected "
            f"{len(EXPECTED_CLASSES)}, received {len(names)}"
        )

    memory_gib = available_memory_gib()
    print("6.1 TensorRT export preflight")
    print(f"  checkpoint    : {model_path}")
    print(f"  output        : {engine_path}")
    print(f"  architecture  : {platform.machine()}")
    print(f"  GPU           : {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch       : {torch.__version__}")
    print(f"  TensorRT      : {trt.__version__}")
    print(f"  classes       : {len(names)}")
    print(f"  input         : 1x3x{EXPECTED_IMAGE_SIZE}x{EXPECTED_IMAGE_SIZE}")
    print(f"  optimization  : FP16, static shapes, NMS disabled")
    print(f"  workspace     : {args.workspace:.1f} GiB")
    if memory_gib is not None:
        print(f"  available RAM : {memory_gib:.2f} GiB")
        if memory_gib < 1.5:
            print(
                "  WARNING: available RAM is low; close VS Code/Chrome and stop "
                "camera/depth applications before exporting."
            )

    if engine_path.exists():
        engine_path.unlink()

    print("\nBuilding the TensorRT engine; this can take several minutes...")
    exported = Path(
        model.export(
            format="engine",
            device=0,
            imgsz=EXPECTED_IMAGE_SIZE,
            batch=1,
            half=True,
            int8=False,
            dynamic=False,
            simplify=True,
            nms=False,
            workspace=args.workspace,
        )
    ).resolve()
    if not exported.is_file():
        fail(f"Ultralytics returned a missing engine path: {exported}")
    if exported != engine_path:
        fail(f"unexpected engine output: expected {engine_path}, got {exported}")

    print(f"\nEngine created: {engine_path}")
    print(f"Engine size   : {engine_path.stat().st_size / (1024 * 1024):.1f} MiB")
    validate_engine_structure(engine_path)

    if not args.skip_inference_check:
        print("Running one inference with the exact 7.0 deployment backend...")
        deployment_dir = SCRIPT_DIR.parent / "7.0_distance_to_haptic_system"
        if not deployment_dir.is_dir():
            fail(f"7.0 deployment project not found: {deployment_dir}")
        sys.path.insert(0, str(deployment_dir))
        from tensorrt_detector import TensorRTDetector

        detector = TensorRTDetector(
            engine_path, confidence=0.45, iou=0.50, max_results=30
        )
        try:
            blank = np.zeros((720, 1280, 3), dtype=np.uint8)
            detector.detect(blank)
        finally:
            detector.close()
        print("7.0 backend inference verification passed.")

    print("\nSUCCESS: use this engine only on this Orin Nano/TensorRT environment.")
    print(engine_path)


if __name__ == "__main__":
    main()
