import asyncio
import os
import sys

import cv2
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.types import FSInputFile
from aiogram.utils.chat_action import ChatActionSender
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import PlateDetector  # noqa: E402
from ocr_engine import LicensePlateReader  # noqa: E402

API_TOKEN = os.getenv("BOT_TOKEN")

TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

if not API_TOKEN:
    exit("Ошибка: BOT_TOKEN не найден в файле .env")

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

detector = PlateDetector()
reader = LicensePlateReader()

@dp.message(Command("start"))
async def start_handler(message: types.Message):
    await message.answer("Привет! Пришли мне фото машины, и я попробую распознать её номер.")

@dp.message(F.photo)
async def handle_photo(message: types.Message):

    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    photo_path = os.path.join(TEMP_DIR, f"temp_{photo.file_id}.jpg")
    await bot.download_file(file_info.file_path, photo_path)

    image = cv2.imread(photo_path)
    await message.answer("Ищу номер...")

    async with ChatActionSender.upload_photo(bot=bot, chat_id=message.chat.id):

        plate_crop = await asyncio.to_thread(detector.find_plate, image)

        if plate_crop is not None:

            debug_path = os.path.join(TEMP_DIR, f"debug_{photo.file_id}.jpg")
            cv2.imwrite(debug_path, plate_crop)

            result_text = await asyncio.to_thread(reader.read_plate, plate_crop)

            if result_text:
                await message.answer(f"✅ Номер найден: `{result_text}`", parse_mode="Markdown")
            else:
                await message.answer("⚠️ Область номера найдена, но текст не распознан.")

            await message.answer_photo(FSInputFile(debug_path), caption="Вот что я увидел")

            os.remove(debug_path)
        else:
            await message.answer("❌ Номер на фото не обнаружен.")

    os.remove(photo_path)

async def main():
    print("Бот запущен и готов к работе")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
