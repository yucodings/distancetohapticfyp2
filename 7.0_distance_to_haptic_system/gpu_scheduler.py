"""Serialize heavy VPI and TensorRT submissions on the shared Jetson GPU."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator


class GpuScheduler:
    """Give stereo a blocking path while detection uses a drop-if-busy path."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @contextmanager
    def depth_job(self) -> Iterator[None]:
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    @contextmanager
    def detection_job(self, *, blocking: bool = False) -> Iterator[bool]:
        acquired = self._lock.acquire(blocking=blocking)
        try:
            yield acquired
        finally:
            if acquired:
                self._lock.release()
