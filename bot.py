import os
import logging
import requests
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import tempfile

# Укажи здесь свой API-ключ от DeepAI
DEEPAI_API_KEY = os.getenv("DEEPAI_API_KEY")

# Telegram Bot Token (будет передан Render через переменные окружения)
BOT_TOKEN = os.getenv("BOT_TOKEN")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start(update: Update, context: CallbackContext):
    update.message.reply_text("Привет! Пришли мне фото, и я улучшу его качество с помощью ИИ 🤖.")

def enhance_image(file_path):
    url = 'https://api.deepai.org/api/torch-srgan'
    with open(file_path, 'rb') as image_file:
        response = requests.post(
            url,
            files={'image': image_file},
            headers={'api-key': DEEPAI_API_KEY}
        )
    data = response.json()
    return data.get('output_url')

def photo_handler(update: Update, context: CallbackContext):
    photo = update.message.photo[-1].get_file()
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tf:
        photo.download(tf.name)
        enhanced_url = enhance_image(tf.name)
        if enhanced_url:
            update.message.reply_photo(photo=enhanced_url)
        else:
            update.message.reply_text("Не удалось улучшить изображение 😢")

def main():
    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, photo_handler))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()