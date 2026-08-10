import os

from fast_plate_ocr import LicensePlateRecognizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "weights", "ocr_best.onnx")
PLATE_CONFIG_PATH = os.path.join(BASE_DIR, "weights", "ocr_plate_config.yaml")


class PlateReader:

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.engine = LicensePlateRecognizer(
            onnx_model_path=MODEL_PATH,
            plate_config_path=PLATE_CONFIG_PATH,
        )

    @staticmethod
    def _get_confidence_level(avg_confidence: float) -> str:
        if avg_confidence >= 0.9:
            return "high"
        elif avg_confidence >= 0.6:
            return "mid"
        else:
            return "low"

    def read_plate(self, image):
        if image is None:
            return None, None

        prediction = self.engine.run_one(image, return_confidence=True)

        plate_text = prediction.plate

        if not plate_text:
            return None, None

        avg_confidence = float(prediction.char_probs.mean())
        confidence_level = self._get_confidence_level(avg_confidence)

        return plate_text, confidence_level
