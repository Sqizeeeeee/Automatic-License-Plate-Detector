import asyncio
import logging

logger = logging.getLogger("PlateDetectorBot.image_processor")

class ImageProcessor:
    def __init__(self, detector, reader):
        self.detector = detector
        self.reader = reader

    async def process_image(self, image):

        try:

            result_text = None
            confidence_level = None

            plate_crop = await asyncio.to_thread(self.detector.find_plate, image)

            if plate_crop is not None:
                result_text, confidence_level = await asyncio.to_thread(self.reader.read_plate,
                                                                        plate_crop)

            return (plate_crop, result_text, confidence_level)

        except Exception as e:
            logger.error(f"Ошибка при обработке картинки: {e}", exc_info=True)
            return (None, None, None)
