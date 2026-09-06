"""Small direct TensorRT runner for the deployed YOLO11 detection engine.

The stereo pipeline must use JetPack's OpenCV build for GStreamer camera
capture. Keeping inference on TensorRT's Python API avoids importing the
Ultralytics/Torch/Matplotlib stack, whose NumPy requirements conflict with
that JetPack OpenCV build.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import DETECTION_INPUT_SIZE
from data_models import Detection


@dataclass(frozen=True)
class EngineMetadata:
    task: str
    batch: int
    image_size: tuple[int, int]
    half: bool
    nms: bool
    names: tuple[str, ...]
    ultralytics_version: str


@dataclass(frozen=True)
class LetterboxTransform:
    ratio: float
    pad_x: float
    pad_y: float
    original_width: int
    original_height: int


def _read_engine_parts(path: Path) -> tuple[dict, bytes]:
    """Return the Ultralytics JSON metadata and the raw TensorRT plan."""
    engine_path = Path(path).expanduser().resolve()
    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
    with engine_path.open("rb") as stream:
        length_bytes = stream.read(4)
        if len(length_bytes) != 4:
            raise RuntimeError(f"TensorRT engine metadata is truncated: {engine_path}")
        metadata_length = int.from_bytes(length_bytes, "little", signed=False)
        if not 2 <= metadata_length <= 1_000_000:
            raise RuntimeError(
                f"Invalid Ultralytics engine metadata length {metadata_length}"
            )
        metadata_bytes = stream.read(metadata_length)
        plan = stream.read()
    try:
        raw = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid TensorRT engine metadata: {error}") from error
    if not plan:
        raise RuntimeError(f"TensorRT plan is missing: {engine_path}")
    return raw, plan


def read_engine_metadata(path: Path) -> EngineMetadata:
    """Read and validate the JSON prefix written by Ultralytics."""
    raw, _ = _read_engine_parts(path)
    args = raw.get("args") or {}
    names_raw = raw.get("names") or {}
    if isinstance(names_raw, dict):
        try:
            ordered_names = tuple(
                str(names_raw[key])
                for key in sorted(names_raw, key=lambda item: int(item))
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError("Engine class IDs must be integer-like") from error
    elif isinstance(names_raw, list):
        ordered_names = tuple(str(item) for item in names_raw)
    else:
        raise RuntimeError("Engine metadata contains no valid class names")

    image_size_raw = raw.get("imgsz")
    if not isinstance(image_size_raw, (list, tuple)) or len(image_size_raw) != 2:
        raise RuntimeError("Engine metadata contains no static two-axis image size")
    metadata = EngineMetadata(
        task=str(raw.get("task", "")),
        batch=int(raw.get("batch", 0)),
        image_size=(int(image_size_raw[0]), int(image_size_raw[1])),
        half=bool(args.get("half", False)),
        nms=bool(args.get("nms", False)),
        names=ordered_names,
        ultralytics_version=str(raw.get("version", "unknown")),
    )
    if metadata.task != "detect":
        raise RuntimeError(f"Expected a detection engine, received {metadata.task!r}")
    if metadata.batch != 1:
        raise RuntimeError(f"Expected batch 1 engine, received {metadata.batch}")
    expected_size = (DETECTION_INPUT_SIZE, DETECTION_INPUT_SIZE)
    if metadata.image_size != expected_size:
        raise RuntimeError(
            f"Expected static {DETECTION_INPUT_SIZE}x{DETECTION_INPUT_SIZE} "
            f"engine, received {metadata.image_size}"
        )
    if not metadata.half:
        raise RuntimeError("Expected an FP16 TensorRT engine")
    if metadata.nms:
        raise RuntimeError("This runner expects raw YOLO output without embedded NMS")
    if not metadata.names:
        raise RuntimeError("TensorRT engine has no classes")
    return metadata


def letterbox_bgr(
    frame: np.ndarray, image_size: tuple[int, int]
) -> tuple[np.ndarray, LetterboxTransform]:
    """Letterbox a BGR frame and return normalized RGB NCHW float32 input."""
    target_height, target_width = image_size
    height, width = frame.shape[:2]
    ratio = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * ratio)))
    resized_height = max(1, int(round(height * ratio)))
    resized = cv2.resize(
        frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    pad_width = target_width - resized_width
    pad_height = target_height - resized_height
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    rgb_nchw = np.ascontiguousarray(
        padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32
    )
    rgb_nchw *= np.float32(1.0 / 255.0)
    transform = LetterboxTransform(
        ratio=ratio,
        pad_x=float(left),
        pad_y=float(top),
        original_width=width,
        original_height=height,
    )
    return rgb_nchw, transform


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Return indices kept by conventional greedy NMS."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        intersection_width = np.maximum(
            0.0,
            np.minimum(x2[current], x2[remaining])
            - np.maximum(x1[current], x1[remaining]),
        )
        intersection_height = np.maximum(
            0.0,
            np.minimum(y2[current], y2[remaining])
            - np.maximum(y1[current], y1[remaining]),
        )
        intersection = intersection_width * intersection_height
        union = areas[current] + areas[remaining] - intersection
        overlap = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0.0,
        )
        order = remaining[overlap <= iou_threshold]
    return kept


def decode_yolo_output(
    output: np.ndarray,
    transform: LetterboxTransform,
    names: tuple[str, ...],
    confidence_threshold: float,
    iou_threshold: float,
    max_results: int,
) -> tuple[Detection, ...]:
    """Decode one YOLO11 ``[1, 4 + classes, anchors]`` output tensor."""
    if output.ndim != 3 or output.shape[0] != 1:
        raise RuntimeError(f"Unexpected YOLO output shape: {output.shape}")
    expected_channels = 4 + len(names)
    if output.shape[1] != expected_channels:
        raise RuntimeError(
            f"Expected {expected_channels} YOLO output channels, got {output.shape[1]}"
        )
    predictions = output[0].T
    class_scores = predictions[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores[np.arange(len(predictions)), class_ids]
    selected = confidences >= confidence_threshold
    if not np.any(selected):
        return ()

    xywh = predictions[selected, :4]
    class_ids = class_ids[selected].astype(np.int32, copy=False)
    confidences = confidences[selected]
    boxes = np.empty((len(xywh), 4), dtype=np.float32)
    boxes[:, 0] = (
        xywh[:, 0] - xywh[:, 2] * 0.5 - transform.pad_x
    ) / transform.ratio
    boxes[:, 1] = (
        xywh[:, 1] - xywh[:, 3] * 0.5 - transform.pad_y
    ) / transform.ratio
    boxes[:, 2] = (
        xywh[:, 0] + xywh[:, 2] * 0.5 - transform.pad_x
    ) / transform.ratio
    boxes[:, 3] = (
        xywh[:, 1] + xywh[:, 3] * 0.5 - transform.pad_y
    ) / transform.ratio
    boxes[:, (0, 2)] = np.clip(
        boxes[:, (0, 2)], 0, transform.original_width - 1
    )
    boxes[:, (1, 3)] = np.clip(
        boxes[:, (1, 3)], 0, transform.original_height - 1
    )

    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, class_ids, confidences = boxes[valid], class_ids[valid], confidences[valid]
    kept: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        local_kept = _nms(boxes[indices], confidences[indices], iou_threshold)
        kept.extend(int(indices[index]) for index in local_kept)
    kept.sort(key=lambda index: float(confidences[index]), reverse=True)

    detections = []
    for index in kept[:max_results]:
        class_id = int(class_ids[index])
        coordinates = tuple(int(round(float(value))) for value in boxes[index])
        detections.append(
            Detection(
                class_id=class_id,
                label=names[class_id],
                confidence=float(confidences[index]),
                bbox=coordinates,
            )
        )
    return tuple(detections)


class TensorRTDetector:
    """Own one TensorRT execution context and reusable CUDA buffers."""

    def __init__(
        self,
        model_path: Path,
        confidence: float,
        iou: float,
        max_results: int,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.metadata = read_engine_metadata(self.model_path)
        if not 0.0 < confidence <= 1.0:
            raise ValueError("Detection confidence must be in (0, 1]")
        if not 0.0 < iou <= 1.0:
            raise ValueError("Detection IoU must be in (0, 1]")
        if max_results < 1:
            raise ValueError("Detection maximum result count must be positive")
        self.confidence = confidence
        self.iou = iou
        self.max_results = max_results
        self.device_label = "TensorRT FP16 cuda:0 (direct runtime)"
        self._device_allocations: list[object] = []
        self._stream = None
        self._device = None
        self._primary_context = None
        self._closed = False

        try:
            import tensorrt as trt
            from cuda.bindings import driver as cuda_driver
        except ImportError as error:
            raise RuntimeError(
                "Direct TensorRT inference requires TensorRT and cuda-python"
            ) from error
        self._trt = trt
        self._cuda_driver = cuda_driver
        try:
            self._driver("initialization", cuda_driver.cuInit(0))
            self._device = self._driver("device selection", cuda_driver.cuDeviceGet(0))
            self._primary_context = self._driver(
                "primary context retention",
                cuda_driver.cuDevicePrimaryCtxRetain(self._device),
            )
            self._driver(
                "context selection", cuda_driver.cuCtxSetCurrent(self._primary_context)
            )
            _, plan = _read_engine_parts(self.model_path)
            self._logger = trt.Logger(trt.Logger.WARNING)
            self._runtime = trt.Runtime(self._logger)
            self._engine = self._runtime.deserialize_cuda_engine(plan)
            if self._engine is None:
                raise RuntimeError("TensorRT could not deserialize the engine plan")
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError("TensorRT could not create an execution context")
            self._configure_io()
        except Exception:
            self.close()
            raise

    def _driver(self, operation: str, result):
        values = result if isinstance(result, tuple) else (result,)
        status = values[0]
        if status != self._cuda_driver.CUresult.CUDA_SUCCESS:
            message = self._cuda_driver.cuGetErrorString(status)
            detail = (
                message[1].decode("utf-8", "replace")
                if len(message) > 1
                else str(status)
            )
            raise RuntimeError(f"CUDA {operation} failed: {detail}")
        if len(values) == 1:
            return None
        if len(values) == 2:
            return values[1]
        return values[1:]

    def _configure_io(self) -> None:
        trt = self._trt
        inputs = []
        outputs = []
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            mode = self._engine.get_tensor_mode(name)
            (inputs if mode == trt.TensorIOMode.INPUT else outputs).append(name)
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"Expected one TensorRT input and output, got {len(inputs)}/{len(outputs)}"
            )
        self._input_name, self._output_name = inputs[0], outputs[0]
        input_shape = tuple(self._engine.get_tensor_shape(self._input_name))
        output_shape = tuple(self._engine.get_tensor_shape(self._output_name))
        expected_input = (1, 3, *self.metadata.image_size)
        expected_output = (1, 4 + len(self.metadata.names), 8400)
        if input_shape != expected_input:
            raise RuntimeError(f"Unexpected TensorRT input shape: {input_shape}")
        if output_shape != expected_output:
            raise RuntimeError(f"Unexpected TensorRT output shape: {output_shape}")
        if self._engine.get_tensor_dtype(self._input_name) != trt.float32:
            raise RuntimeError("Expected TensorRT float32 input binding")
        if self._engine.get_tensor_dtype(self._output_name) != trt.float32:
            raise RuntimeError("Expected TensorRT float32 output binding")

        self._host_input = np.empty(input_shape, dtype=np.float32)
        self._host_output = np.empty(output_shape, dtype=np.float32)
        self._stream = self._driver(
            "stream creation", self._cuda_driver.cuStreamCreate(0)
        )
        for array in (self._host_input, self._host_output):
            allocation = self._driver(
                "allocation", self._cuda_driver.cuMemAlloc(array.nbytes)
            )
            self._device_allocations.append(allocation)
        if not self._context.set_tensor_address(
            self._input_name, int(self._device_allocations[0])
        ):
            raise RuntimeError("TensorRT rejected the input tensor address")
        if not self._context.set_tensor_address(
            self._output_name, int(self._device_allocations[1])
        ):
            raise RuntimeError("TensorRT rejected the output tensor address")

    def warmup(self) -> None:
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.detect(blank)

    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        if self._closed:
            raise RuntimeError("TensorRT detector is closed")
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Detector input must be an HxWx3 uint8 BGR image")
        prepared, transform = letterbox_bgr(frame, self.metadata.image_size)
        np.copyto(self._host_input, prepared)
        cuda_driver = self._cuda_driver
        self._driver(
            "input copy",
            cuda_driver.cuMemcpyHtoDAsync(
                self._device_allocations[0],
                self._host_input.ctypes.data,
                self._host_input.nbytes,
                self._stream,
            ),
        )
        if not self._context.execute_async_v3(stream_handle=int(self._stream)):
            raise RuntimeError("TensorRT inference execution failed")
        self._driver(
            "output copy",
            cuda_driver.cuMemcpyDtoHAsync(
                self._host_output.ctypes.data,
                self._device_allocations[1],
                self._host_output.nbytes,
                self._stream,
            ),
        )
        self._driver(
            "stream synchronization", cuda_driver.cuStreamSynchronize(self._stream)
        )
        return decode_yolo_output(
            self._host_output,
            transform,
            self.metadata.names,
            self.confidence,
            self.iou,
            self.max_results,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        cuda_driver = getattr(self, "_cuda_driver", None)
        if cuda_driver is not None:
            for allocation in reversed(self._device_allocations):
                try:
                    cuda_driver.cuMemFree(allocation)
                except Exception:
                    pass
            self._device_allocations.clear()
            if self._stream is not None:
                try:
                    cuda_driver.cuStreamDestroy(self._stream)
                except Exception:
                    pass
                self._stream = None
        self._context = None
        self._engine = None
        self._runtime = None
        if cuda_driver is not None and self._device is not None:
            try:
                cuda_driver.cuDevicePrimaryCtxRelease(self._device)
            except Exception:
                pass
            self._device = None
            self._primary_context = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
