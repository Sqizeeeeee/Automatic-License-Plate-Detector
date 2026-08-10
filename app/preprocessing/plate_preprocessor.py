import cv2
import numpy as np


class PlatePreprocessor:

    def __init__(self,
             target_width=300,
             max_upscale_factor=6.0,
             clahe_clip_limit=2.0,
             clahe_tile_size=8,
             sharpen_amount=1.5,
             sharpen_blur_sigma=3):
        self.target_width = target_width
        self.max_upscale_factor = max_upscale_factor
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.sharpen_amount = sharpen_amount
        self.sharpen_blur_sigma = sharpen_blur_sigma

    def process(self, image):

        if image is None or image.size == 0:
            return image

        _, width = image.shape[:2]

        upscale_factor = self._get_upscale_factor(width)

        upscaled_image = self._upscale(image, upscale_factor)

        clahe_image = self._apply_clahe(upscaled_image)

        sharpened_image = self._sharpen(clahe_image)

        return sharpened_image


    def _get_upscale_factor(self, width) -> float:
        """
        Определяет коэффициент апскейла в зависимости от ширины кропа.
        Чем меньше исходник — тем больше коэффициент.
        """
        if width <= 0:
            return 1.0

        upscale_factor = self.target_width / width

        return min(upscale_factor, self.max_upscale_factor)

    def _upscale(self, image, factor) -> np.ndarray:
        """
        Ресайзит image с заданным коэффициентом через cv2.resize, interpolation=cv2.INTER_CUBIC.
        """
        result_image = cv2.resize(image,
                                  None,
                                  fx=factor,
                                  fy=factor,
                                  interpolation=cv2.INTER_CUBIC)

        return result_image

    def _apply_clahe(self, image) -> np.ndarray:
        """
        Применяет CLAHE для выравнивания локального контраста.
        """
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        l_channel, a_channel, b_channel = cv2.split(lab_image)

        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit,
                                tileGridSize=(self.clahe_tile_size, self.clahe_tile_size))

        l_enhanced = clahe.apply(l_channel)

        lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))

        result_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_Lab2BGR)

        return result_image

    def _sharpen(self, image):
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=self.sharpen_blur_sigma)
        sharpened = cv2.addWeighted(
            image, self.sharpen_amount,
            blurred, -(self.sharpen_amount - 1),
            0
        )
        return sharpened
