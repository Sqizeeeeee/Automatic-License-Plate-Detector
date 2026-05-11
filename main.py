import cv2
from detector import PlateDetector
from ocr_engine import LicensePlateReader

def main():

    detector = PlateDetector()
    reader = LicensePlateReader()

    img_path = 'images/image.png' 
    image = cv2.imread(img_path)

    if image is None:
        print("Ошибка: Файл не найден")
        return
    print("Ищем номер на фото...")
    plate_crop = detector.find_plate(image)

    if plate_crop is not None:
        print("Область найдена! Распознаем текст...")
        results = reader.read_plate(plate_crop)
        
        print(f"Результат: {results}")
        
        cv2.imwrite('images/debug_plate.jpg', plate_crop)
        print("Вырезанный номер сохранен в images/debug_plate.jpg")
    else:
        print("Детектор не смог найти прямоугольник номера.")

if __name__ == "__main__":
    main()
