import numpy as np
from rapidocr_onnxruntime import RapidOCR


class PlateReader:
    def __init__(self, preprocessor):
        self.engine = RapidOCR()
        self.preprocessor = preprocessor

    @staticmethod
    def _get_box_center_y(box_coordinates) -> float:
        summ = 0
        for box in box_coordinates:
            summ += box[-1]
        return summ / 4

    @staticmethod
    def _get_box_height(box_coordinates) -> float:
        ys = [point[1] for point in box_coordinates]
        return max(ys) - min(ys)

    @staticmethod
    def _get_box_center_x(box_coordinates) -> float:
        xs = [point[0] for point in box_coordinates]
        return sum(xs) / len(xs)

    @staticmethod
    def _get_confidence_level(avg_confidence: float) -> str:
        if avg_confidence >= 0.9:
            return "high"
        elif avg_confidence >= 0.6:
            return "mid"
        else:
            return "low"

    def _filter_chunks(self, results):
        if not results:
            return []

        heights = [self._get_box_height(res[0]) for res in results]
        max_height = max(heights)

        height_filtered = [
            res for res in results
            if self._get_box_height(res[0]) >= max_height * 0.6
        ]

        if not height_filtered:
            return []

        center_ys = [self._get_box_center_y(res[0]) for res in height_filtered]
        median_center_y = np.median(center_ys)

        remaining_heights = [self._get_box_height(res[0]) for res in height_filtered]
        avg_height = sum(remaining_heights) / len(remaining_heights)
        max_deviation = avg_height * 0.7

        valid_chunks = []

        for coords, text, confidence in height_filtered:

            if confidence < 0.5:
                continue

            current_center_y = self._get_box_center_y(coords)
            if abs(current_center_y - median_center_y) > max_deviation:
                continue

            # фильтруем только ASCII алфанумерик — отсекаем иероглифы и прочий мусор
            clean_text = "".join(c for c in text if c.isascii() and c.isalnum()).upper()
            if not clean_text:
                continue

            valid_chunks.append((coords, clean_text, confidence))

        return valid_chunks

    def _assemble_text(self, valid_chunks):
        if not valid_chunks:
            return None, None

        sorted_chunks = sorted(
            valid_chunks,
            key=lambda chunk: self._get_box_center_x(chunk[0])
        )

        final_text = "".join(chunk[1] for chunk in sorted_chunks)

        if len(final_text) <= 2:
            return None, None

        total_weight = sum(len(text) for _, text, _ in sorted_chunks)
        weighted_sum = sum(len(text) * conf for _, text, conf in sorted_chunks)
        avg_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0

        confidence_level = self._get_confidence_level(avg_confidence)

        return final_text, confidence_level

    def read_plate(self, image):
        if image is None:
            return None, None

        processed = self.preprocessor.process(image)

        chunks, _ = self.engine(processed)

        if not chunks:
            return None, None

        filtered = self._filter_chunks(chunks)
        result_text, confidence_level = self._assemble_text(filtered)

        return result_text, confidence_level
