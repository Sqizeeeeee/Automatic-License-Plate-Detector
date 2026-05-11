import os
import asyncio
import sys
import cv2
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import FSInputFile
from dotenv import load_dotenv


load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import PlateDetector
from ocr_engine import LicensePlateReader


API_TOKEN = os.getenv("BOT_TOKEN")

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
    photo_path = f"temp_{photo.file_id}.jpg"
    await bot.download_file(file_info.file_path, photo_path)

    image = cv2.imread(photo_path)
    await message.answer("Ищу номер...")
    
    plate_crop = detector.find_plate(image)
    
    if plate_crop is not None:

        debug_path = f"debug_{photo.file_id}.jpg"
        cv2.imwrite(debug_path, plate_crop)
        
        result_text = reader.read_plate(plate_crop)
        
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
