import os

from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Переключатель между лёгкой (nano) и более точной (small) версией модели.
USE_LIGHT_MODEL = True

NANO_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "yolov8n.pt")
SMALL_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "yolov8s.pt")

VEHICLE_CLASS_IDS = {2, 5, 7}  # car, bus, truck


class VehicleDetector:

    def __init__(self, use_light_model: bool = USE_LIGHT_MODEL, conf: float = 0.4):
        weights_path = NANO_WEIGHTS_PATH if use_light_model else SMALL_WEIGHTS_PATH
        self.model = YOLO(weights_path)
        self.conf = conf

    def find_vehicles(self, image):
        """
        Возвращает список боксов (x1, y1, x2, y2) всех найденных на кадре
        транспортных средств (car, bus, truck). Пустой список, если
        ничего не найдено.
        """
        results = self.model.predict(
            image,
            conf=self.conf,
            classes=list(VEHICLE_CLASS_IDS),
            verbose=False,
        )

        boxes = []
        for result in results:
            for box in result.boxes:
                coords = box.xyxy.flatten().tolist()
                x1, y1, x2, y2 = map(int, coords)
                boxes.append((x1, y1, x2, y2))

        return boxes