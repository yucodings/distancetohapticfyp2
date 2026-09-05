"""Reusable widgets for the distance-to-haptic dashboard."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout

from config import DISPLAY_LABELS, FRAME_ASPECT_RATIO


class AspectRatioVideoLabel(QLabel):
    def __init__(self, waiting_text: str):
        super().__init__(waiting_text)
        self._pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setObjectName("videoFrame")
        self.setMinimumSize(240, 135)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return int(width / FRAME_ASPECT_RATIO)

    def minimumSizeHint(self) -> QSize:
        return QSize(240, 135)

    def sizeHint(self) -> QSize:
        return QSize(640, 360)

    def set_frame(self, image: QImage) -> None:
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            return
        self._pixmap = pixmap
        self._rescale()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class VideoPanel(QFrame):
    def __init__(self, title: str, waiting_text: str):
        super().__init__()
        self.setObjectName("panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)
        title_label = QLabel(title)
        title_label.setObjectName("panelTitle")
        self.video = AspectRatioVideoLabel(waiting_text)
        layout.addWidget(title_label)
        layout.addWidget(self.video, 1)

    def set_image(self, image: QImage) -> None:
        self.video.set_frame(image)


class ZoneCard(QFrame):
    """A deliberately minimal card: zone name plus metre value only."""

    def __init__(self, zone_key: str):
        super().__init__()
        self.zone_key = zone_key
        self.setObjectName("zoneCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(3)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        name = QLabel(DISPLAY_LABELS[zone_key])
        name.setObjectName("zoneName")
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value = QLabel("-- m")
        self.value.setObjectName("zoneValue")
        self.value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(name)
        layout.addWidget(self.value)

    def set_distance(self, distance_m: Optional[float]) -> None:
        self.value.setText("-- m" if distance_m is None else f"{distance_m:.2f} m")


class StatusBadge(QLabel):
    def __init__(self, prefix: str):
        super().__init__(f"{prefix}: Stopped")
        self.prefix = prefix
        self.setObjectName("statusBadge")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_status(self, status: str) -> None:
        self.setText(f"{self.prefix}: {status}")
        normalized = status.lower()
        if any(word in normalized for word in ("connected", "ready", "enabled")):
            colors = "background:#dcfce7; color:#166534;"
        elif any(word in normalized for word in ("starting", "loading")):
            colors = "background:#dbeafe; color:#1d4ed8;"
        elif any(word in normalized for word in ("error", "emergency")):
            colors = "background:#fee2e2; color:#b91c1c;"
        else:
            colors = "background:#e2e8f0; color:#334155;"
        self.setStyleSheet(
            "QLabel#statusBadge { border-radius:11px; padding:6px 9px; "
            f"font-weight:700; {colors} }}"
        )

