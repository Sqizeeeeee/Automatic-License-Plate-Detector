import asyncio
import os

import cv2
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.types import FSInputFile
from aiogram.utils.chat_action import ChatActionSender
from detector import PlateDetector
from dotenv import load_dotenv
from image_processor import ImageProcessor
from ocr_engine import PlateReader
from preprocessor import PlatePreprocessor
from utils import setup_logger
from video_processor import VideoProcessor

load_dotenv()

API_TOKEN = os.getenv("BOT_TOKEN")

TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

logger = setup_logger()

if not API_TOKEN:
    exit("Ошибка: BOT_TOKEN не найден в файле .env")

bot = Bot(token=API_TOKEN)
dp = Dispatcher()
preprocessor = PlatePreprocessor()
detector = PlateDetector()
reader = PlateReader(preprocessor)
image_processor = ImageProcessor(detector, reader)
video_processor = VideoProcessor(detector, reader)


@dp.message(Command("start"))
async def start_handler(message: types.Message):
    await message.answer("Привет! Пришли мне фото/видео машины, и я попробую распознать её номер.")

@dp.message(F.photo)
async def photo_hadler(message: types.Message):

    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    photo_path = os.path.join(TEMP_DIR, f"temp_{photo.file_id}.jpg")
    await bot.download_file(file_info.file_path, photo_path)

    debug_path = None

    try:
        image = cv2.imread(photo_path)
        await message.answer("Ищу номер...")

        async with ChatActionSender.upload_photo(bot=bot, chat_id=message.chat.id):

            result_plate, result_text, confidence_level = await image_processor.process_image(image)

            if result_plate is not None:
                h, w = result_plate.shape[:2]

                if h < 10 or w < 10:
                    await message.answer("Область номера найдена, но кроп " \
                    "получился слишком маленьким для отправки.")
                else:
                    debug_path = os.path.join(TEMP_DIR, f"debug_{photo.file_id}.jpg")
                    cv2.imwrite(debug_path, result_plate)

                    if result_text:
                        confidence_labels = {
                            "high": "✅ высокая уверенность",
                            "mid": "⚠️ средняя уверенность",
                            "low": "❗ низкая уверенность"
                        }
                        label = confidence_labels.get(confidence_level, "")
                        await message.answer(
                            f"Номер найден: `{result_text}`\n{label}",
                            parse_mode="Markdown"
                        )

                    elif not result_text:
                        await message.answer("Область номера найдена, но текст не распознан.")

                    await message.answer_photo(FSInputFile(debug_path), caption="Вот что я увидел")

            else:
                await message.answer("Номер на фото не обнаружен.")

    except Exception as e:
        logger.error(
            f"Ошибка при обработке фотографии от пользователя {message.from_user.id}: {e}",
            exc_info=True,
        )
        await message.answer(
            "Произошла непредвиденная ошибка при обработке изображения."
        )

    finally:
        if debug_path and os.path.exists(debug_path):
            os.remove(debug_path)
        if os.path.exists(photo_path):
            os.remove(photo_path)

@dp.message(F.video)
async def video_hadler(message: types.Message):

    video = message.video

    file_info = await bot.get_file(video.file_id)
    video_path = os.path.join(TEMP_DIR, f"temp_{video.file_id}")

    await bot.download_file(file_info.file_path, video_path)

    try:

        all_plates = await video_processor.process_video(video_path)

        if all_plates:

            result = "Вот что я нашел: "

            for plate_text, confidence_level in all_plates:
                confidence_labels = {
                    "high": "✅",
                    "mid": "⚠️",
                    "low": "❗"
                }
                label = confidence_labels.get(confidence_level, "")
                result += f"{plate_text} {label} "

            await message.answer(result)

        else:
            await message.answer("Номер на видео не обнаружен.")

    except Exception as e:
        logger.error(
            f"Ошибка при обработке видео от пользователя {message.from_user.id}: {e}",
            exc_info=True,
        )
        await message.answer(
            "Произошла непредвиденная ошибка при обработке видео."
        )

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

async def main():
    logger.info("Бот запущен и готов к работе")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
