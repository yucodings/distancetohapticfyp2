from ultralytics import YOLO

model = YOLO(
    "/media/orin_nano/ESD-USB/FYP model/best.engine",
    task="detect"
)

model.predict(
    source=0,
    imgsz=640,
    conf=0.25,
    device=0,
    show=True
)
