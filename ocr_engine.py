import cv2
from rapidocr_onnxruntime import RapidOCR


class LicensePlateReader:
    def __init__(self):
        self.engine = RapidOCR()

    @staticmethod
    def _get_box_area(box_coordinates):
        max_h = 0
        max_w = 0
        for first_coord in range(len(box_coordinates)):
            for seckond_coord in range(first_coord + 1, len(box_coordinates)):
                coord1 = box_coordinates[first_coord]
                coord2 = box_coordinates[seckond_coord]
                max_w = max(max_w, abs(coord1[0] - coord2[0]))
                max_h = max(max_h, abs(coord1[1] - coord2[1]))

        return int(max_h * max_w)

    def read_plate(self, image):
        if image is None:
            return None

        pad = 20
        image_padded = cv2.copyMakeBorder(
            image, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )

        img = cv2.resize(image_padded, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        results, _ = self.engine(img)

        if not results:
            return None

        max_area = 0
        for item in results:
            max_area = max(max_area, self._get_box_area(item[0]))

        valid_chunks = []

        for res in results:
            coords = res[0]
            text = str(res[1])

            clean_text = "".join(filter(str.isalnum, text)).upper()

            current_area = self._get_box_area(coords)

            if (len(clean_text) > 2 and len(clean_text) <= 10 and current_area >= max_area*0.2):
                valid_chunks.append(clean_text)

        final_text = "".join(valid_chunks)

        return final_text if len(final_text) > 2 else None
