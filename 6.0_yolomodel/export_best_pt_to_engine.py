#!/usr/bin/env python3
"""Export the trained YOLO11n best.pt checkpoint to a Jetson TensorRT engine.

Run this script on the same Jetson that will use the resulting engine. TensorRT
engine files are tied to the device architecture and installed TensorRT version.
"""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

# Export and inference do not need a GUI matplotlib backend.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import tensorrt as trt
import torch
from ultralytics import YOLO


DEFAULT_MODEL = Path("/home/orin_nano/Desktop/FYP2/6.0_yolomodel/best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Trained Ultralytics checkpoint (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Static square input size; must match deployment preprocessing.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Static inference batch size (batch 1 is best for live camera latency).",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Export FP32 instead of the recommended FP16 engine.",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=None,
        help="TensorRT workspace limit in GiB; default lets Ultralytics choose.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing engine with the same name.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the one-frame engine inference test after export.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    expected_engine = model_path.with_suffix(".engine")

    if not model_path.is_file():
        fail(f"checkpoint not found: {model_path}")
    if model_path.suffix.lower() != ".pt":
        fail(f"expected a .pt checkpoint, received: {model_path.name}")
    if args.imgsz <= 0 or args.batch <= 0:
        fail("--imgsz and --batch must be positive integers")
    if expected_engine.exists() and not args.force:
        fail(f"engine already exists: {expected_engine} (use --force to replace it)")

    print("Jetson TensorRT export preflight")
    print(f"  Device architecture : {platform.machine()}")
    print(f"  PyTorch             : {torch.__version__}")
    print(f"  TensorRT            : {trt.__version__}")
    print(f"  CUDA available      : {torch.cuda.is_available()}")
    if platform.machine() != "aarch64":
        print("  WARNING: This does not look like a Jetson/aarch64 environment.")
    if not torch.cuda.is_available():
        fail(
            "PyTorch cannot access CUDA. TensorRT export must run in the Jetson "
            "environment with a working NVIDIA driver, CUDA, and TensorRT."
        )
    print(f"  CUDA device         : {torch.cuda.get_device_name(0)}")
    print(f"  Source checkpoint   : {model_path}")
    print(f"  Precision           : {'FP32' if args.fp32 else 'FP16'}")
    print(f"  Input               : {args.batch} x 3 x {args.imgsz} x {args.imgsz}")

    model = YOLO(str(model_path), task="detect")
    if model.task != "detect":
        fail(f"checkpoint task is {model.task!r}; this script expects object detection")
    names = model.names
    print(f"  Classes             : {len(names)}")
    print("  Class names         : " + ", ".join(str(names[index]) for index in names))

    if expected_engine.exists():
        expected_engine.unlink()

    export_options = {
        "format": "engine",
        "device": 0,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "half": not args.fp32,
        "int8": False,
        "dynamic": False,
        "simplify": True,
        "nms": False,
    }
    if args.workspace is not None:
        export_options["workspace"] = args.workspace

    print("\nBuilding TensorRT engine. This can take several minutes...")
    exported = Path(model.export(**export_options)).resolve()
    if not exported.is_file():
        fail(f"Ultralytics reported an export, but no engine exists at {exported}")

    print(f"\nEngine created: {exported}")
    print(f"Engine size   : {exported.stat().st_size / (1024 * 1024):.1f} MiB")

    if not args.skip_verify:
        print("Verifying the engine with one blank frame...")
        engine_model = YOLO(str(exported), task="detect")
        blank_frame = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
        results = engine_model.predict(
            source=blank_frame,
            imgsz=args.imgsz,
            device=0,
            verbose=False,
        )
        if not results:
            fail("engine verification returned no result object")
        if len(results[0].names) != len(names):
            fail(
                "engine verification found a class-count mismatch: "
                f"PT={len(names)}, engine={len(results[0].names)}"
            )
        print(f"Verification passed: {len(results[0].names)} classes preserved.")

    print("\nUse this .engine file only on this Jetson/TensorRT environment.")


if __name__ == "__main__":
    main()
