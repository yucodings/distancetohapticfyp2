from typing import Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout

from config import DISPLAY_LABELS, FRAME_ASPECT_RATIO
from data_models import DetectionResult, ZoneResult


class AspectRatioVideoLabel(QLabel):
    def __init__(self, aspect_ratio: float = FRAME_ASPECT_RATIO):
        super().__init__("Waiting for stream...")
        self.aspect_ratio = aspect_ratio
        self._pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setObjectName("videoFrame")
        self.setMinimumSize(240, 135)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width: int):
        return int(width / self.aspect_ratio)

    def minimumSizeHint(self):
        # Capture resolution must not become the widget's minimum window size.
        return QSize(240, 135)

    def sizeHint(self):
        return QSize(640, 360)

    def set_frame(self, image: QImage):
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            return
        self._pixmap = pixmap
        self._update_scaled_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._pixmap is None or self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)


class VideoPanel(QFrame):
    def __init__(self, title: str, subtitle: str):
        super().__init__()
        self.setObjectName("panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("panelTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("panelSubtitle")

        self.video_label = AspectRatioVideoLabel()

        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addWidget(self.video_label)

    def set_image(self, image: QImage):
        self.video_label.set_frame(image)


class ZoneCard(QFrame):
    def __init__(self, zone_name: str):
        super().__init__()
        self.zone_name = zone_name
        self.setObjectName("zoneCard")
        self.setMinimumHeight(104)
        self.setMaximumHeight(116)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(3)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.name_label = QLabel(DISPLAY_LABELS[zone_name])
        self.name_label.setObjectName("zoneName")
        self.value_label = QLabel("Stereo nearest: --")
        self.value_label.setObjectName("zoneValue")
        self.object_label = QLabel("YOLO object: none")
        self.object_label.setObjectName("zoneObject")

        layout.addWidget(self.name_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.object_label)

        self.set_tone("invalid")

    def set_tone(self, tone: str):
        style_map = {
            "clear": "background:#f8fafc; border:1px solid #d7dee7;",
            "green": "background:#ecfdf5; border:1px solid #a7f3d0;",
            "orange": "background:#fff7ed; border:1px solid #fdba74;",
            "red": "background:#fef2f2; border:1px solid #fca5a5;",
            "invalid": "background:#f8fafc; border:1px solid #d7dee7;",
        }
        self.setStyleSheet(
            "QFrame#zoneCard {"
            f"{style_map.get(tone, style_map['invalid'])} border-radius:14px;"
            "}"
        )

    def update_zone(self, result: ZoneResult):
        if result.depth_m is None:
            self.value_label.setText("Stereo nearest: --")
            self.set_tone("invalid")
            return

        self.value_label.setText(f"Stereo nearest: {result.depth_m:.2f} m")
        self.set_tone(result.tone)

    def update_detection(self, detection: Optional[DetectionResult]):
        if detection is None:
            self.object_label.setText("YOLO object: none")
            return

        distance = "-- m" if detection.depth_m is None else f"{detection.depth_m:.2f} m"
        self.object_label.setText(
            f"YOLO object: {detection.label} · {detection.confidence:.0%} · {distance}"
        )
