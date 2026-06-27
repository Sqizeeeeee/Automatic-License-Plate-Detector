import cv2
from rapidocr_onnxruntime import RapidOCR


class LicensePlateReader:
    def __init__(self):
        self.engine = RapidOCR()

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

        full_text = ""
        for res in results:
            full_text += str(res[1])

        clean_text = "".join(filter(str.isalnum, full_text)).upper()

        return clean_text if len(clean_text) > 2 else None
