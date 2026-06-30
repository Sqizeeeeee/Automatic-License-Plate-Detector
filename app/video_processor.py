import asyncio
import logging

import cv2

logger = logging.getLogger("PlateDetectorBot.video_processor")

class VideoProcessor:
    def __init__(self, detector, reader, frame_skip=5):
        self.detector = detector
        self.reader = reader
        self.frame_skip = frame_skip

    async def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        plates_seen = {}
        all_plates = set()

        try:

            while cap.isOpened():

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if frame_count % self.frame_skip != 0:
                    continue

                plate_crop = await asyncio.to_thread(self.detector.find_plate, frame)

                if plate_crop is not None:

                    result_text, confidence_level = await asyncio.to_thread(self.reader.read_plate,
                                                                            plate_crop)

                    if result_text and confidence_level:
                        if result_text not in plates_seen:
                            plates_seen[result_text] = []
                        plates_seen[result_text].append(confidence_level)

            for text, levels in plates_seen.items():
                final_level = max(set(levels), key=levels.count)
                all_plates.add((text, final_level))

            return all_plates

        except Exception as e:
            logger.error(f"Ошибка при обработке видео: {e}", exc_info=True)
            return all_plates

        finally:
            cap.release()
