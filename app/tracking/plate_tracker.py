"""
Трекинг одной машины между кадрами видео на основе пересечения боксов (IoU).
"""


def compute_iou(box_a, box_b) -> float:
    """
    Intersection over Union для двух боксов в формате (x1, y1, x2, y2).
    Возвращает значение от 0 (боксы не пересекаются) до 1 (идентичны).
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


class VehicleTrack:
    """
    Представляет одну машину, отслеживаемую на протяжении нескольких кадров.
    Копит прочитанные тексты номера вместе с их уровнем уверенности, пока машина остаётся в кадре.
    """

    def __init__(self, bbox):
        self.bbox = bbox
        self.missed_frames = 0
        self._text_readings: dict[str, list[str]] = {}

    def update_bbox(self, bbox):
        self.bbox = bbox
        self.missed_frames = 0

    def add_reading(self, text: str, confidence_level: str):
        self._text_readings.setdefault(text, []).append(confidence_level)

    def finalize(self, min_occurrences: int) -> tuple[str, str] | None:
        """
        Возвращает (text, confidence_level) для самого часто читаемого
        текста, если он набрал не меньше min_occurrences подтверждений.
        """
        candidates = {
            text: levels
            for text, levels in self._text_readings.items()
            if len(levels) >= min_occurrences
        }

        if not candidates:
            return None

        best_text = max(candidates, key=lambda text: len(candidates[text]))
        levels = candidates[best_text]
        final_level = max(set(levels), key=levels.count)

        return best_text, final_level
