import cv2
import numpy as np
from ultralytics import YOLO


class PlateDetector:
    def __init__(self):
        self.model = YOLO('best.pt')

    def find_plate(self, image):
        results = self.model.predict(image, conf=0.5, verbose=False)

        for result in results:
            for box in result.boxes:
                coords = box.xyxy.flatten().tolist()
                x1, y1, x2, y2 = map(int, coords)

                pad = 5
                h_orig, w_orig = image.shape[:2]
                raw_plate = image[max(0, y1-pad):min(h_orig, y2+pad),
                                  max(0, x1-pad):min(w_orig, x2+pad)]

                corrected_plate = self.deskew_plate(raw_plate)
                return corrected_plate
        return None

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def deskew_plate(self, plate):
        if plate is None or plate.size == 0:
            return plate

        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return plate

        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.array(box).astype(int)

        ordered_box = self.order_points(box)
        (tl, tr, br, bl) = ordered_box

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        _, w_plate = plate.shape[:2]

        if (maxWidth < 0.6 * w_plate) or maxHeight == 0:
            return plate

        aspect_ratio = maxWidth / maxHeight

        if not (1.5 < aspect_ratio < 6):
            return plate

        dst_pts = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(ordered_box, dst_pts)
        warped = cv2.warpPerspective(plate, M, (maxWidth, maxHeight))

        return warped
