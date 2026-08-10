import asyncio
import logging

import cv2
from tracking.plate_tracker import VehicleTrack, compute_iou

logger = logging.getLogger("PlateDetectorBot.video_processor")


class VideoProcessor:
    def __init__(
        self,
        detector,
        reader,
        frame_skip=5,
        min_occurrences=3,
        iou_threshold=0.3,
        max_missed_frames=2,
    ):
        self.detector = detector
        self.reader = reader
        self.frame_skip = frame_skip
        self.min_occurrences = min_occurrences
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames

    async def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        current_track: VehicleTrack | None = None
        confirmed_plates = set()

        try:
            while cap.isOpened():

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if frame_count % self.frame_skip != 0:
                    continue

                detection = await asyncio.to_thread(
                    self.detector.find_plate_with_bbox, frame
                )

                if detection is None:
                    current_track = self._handle_missed_frame(
                        current_track, confirmed_plates
                    )
                    continue

                plate_crop, bbox = detection

                same_vehicle = (
                    current_track is not None
                    and compute_iou(current_track.bbox, bbox) >= self.iou_threshold
                )

                if not same_vehicle:
                    self._finalize_track(current_track, confirmed_plates)
                    current_track = VehicleTrack(bbox)
                else:
                    current_track.update_bbox(bbox)

                result_text, confidence_level = await asyncio.to_thread(
                    self.reader.read_plate, plate_crop
                )

                if result_text and confidence_level:
                    current_track.add_reading(result_text, confidence_level)

            self._finalize_track(current_track, confirmed_plates)

            return confirmed_plates

        except Exception as e:
            logger.error(f"Ошибка при обработке видео: {e}", exc_info=True)
            return confirmed_plates

        finally:
            cap.release()

    def _handle_missed_frame(self, current_track, confirmed_plates):
        """
        Вызывается, когда на обработанном кадре машина не найдена.
        Если трек отсутствовал в кадре достаточно долго - считаем,
        что машина уехала, и финализируем трек.
        """
        if current_track is None:
            return None

        current_track.missed_frames += 1

        if current_track.missed_frames > self.max_missed_frames:
            self._finalize_track(current_track, confirmed_plates)
            return None

        return current_track

    def _finalize_track(self, track, confirmed_plates):
        if track is None:
            return

        result = track.finalize(self.min_occurrences)
        if result is not None:
            confirmed_plates.add(result)
