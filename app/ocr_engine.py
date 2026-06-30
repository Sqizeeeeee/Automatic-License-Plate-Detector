import numpy as np
from rapidocr_onnxruntime import RapidOCR


class PlateReader:
    def __init__(self, preprocessor):
        self.engine = RapidOCR()
        self.preprocessor = preprocessor

    # @staticmethod
    # def _get_box_area(box_coordinates) -> int:
    #     max_h = 0
    #     max_w = 0
    #     for first_coord in range(len(box_coordinates)):
    #         for seckond_coord in range(first_coord + 1, len(box_coordinates)):
    #             coord1 = box_coordinates[first_coord]
    #             coord2 = box_coordinates[seckond_coord]
    #             max_w = max(max_w, abs(coord1[0] - coord2[0]))
    #             max_h = max(max_h, abs(coord1[1] - coord2[1]))

    #     return int(max_h * max_w)

    @staticmethod
    def _get_box_center_y(box_coordinates) -> float:

        summ = 0

        for box in box_coordinates:
            summ += box[-1]

        return summ / 4

    @staticmethod
    def _get_box_height(box_coordinates):
        ys = [point[1] for point in box_coordinates]
        return max(ys) - min(ys)

    @staticmethod
    def _get_box_center_x(box_coordinates):
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

        # Проход 1: фильтр по высоте (отсекаем мелкий контекстный текст)
        heights = [self._get_box_height(res[0]) for res in results]
        max_height = max(heights)

        height_filtered = [
            res for res in results
            if self._get_box_height(res[0]) >= max_height * 0.6
        ]

        if not height_filtered:
            return []

        # Проход 2: фильтр по y-выравниванию (отсекаем выбросы среди оставшихся)
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

            clean_text = "".join(filter(str.isalnum, text)).upper()
            if not clean_text:
                continue

            valid_chunks.append((coords, clean_text, confidence))

        return valid_chunks

    def _assemble_text(self, valid_chunks):
        # valid_chunks - список кортежей (coords, clean_text, confidence)

        sorted_chunks = sorted(valid_chunks, key=lambda chunk: self._get_box_center_x(chunk[0]))

        final_text = "".join(chunk[1] for chunk in sorted_chunks)

        if len(final_text) <= 2:
            return None, None

        total_weight = sum(len(text) for _, text, conf in sorted_chunks)
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

        chunks = self._filter_chunks(chunks)
        result_text, confidence_level = self._assemble_text(chunks)

        return result_text, confidence_level

    def read_plate(self, image):
        if image is None:
            return None

        processed = self.preprocessor.process(image)

        chunks, _ = self.engine(processed)

        print("RAW chunks:", chunks)

        if not chunks:
            return None

        chunks = self._filter_chunks(chunks)
        print("FILTERED chunks:", chunks)
        result = self._assemble_text(chunks)


        return result
