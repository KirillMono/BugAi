import os
import logging
import urllib.request
import tempfile

import cv2
import torch
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv('BOT_TOKEN')


def download_file(url, path):
    if not os.path.exists(path):
        logger.info(f"Скачиваю {path}...")
        urllib.request.urlretrieve(url, path)
        logger.info(f"{path} скачан.")
    else:
        logger.info(f"{path} уже есть.")


def download_models():
    download_file(
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus.pth',
        'RealESRGAN_x4plus.pth'
    )
    download_file(
        'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        'GFPGANv1.3.pth'
    )


def init_models():
    download_models()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_growth=32)
    upsampler = RealESRGANer(
        scale=4,
        model_path='RealESRGAN_x4plus.pth',
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device
    )
    face_enhancer = GFPGANer(
        model_path='GFPGANv1.3.pth',
        upscale=4,
        arch='clean',
        channel_multiplier=2,
        device=device
    )
    return upsampler, face_enhancer


upsampler, face_enhancer = None, None


def start(update: Update, context: CallbackContext):
    update.message.reply_text('Привет! Пришли мне фото, и я улучшу его качество.')


def enhance_image(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        return False
    output, _ = upsampler.enhance(img, outscale=4)
    _, _, output = face_enhancer.enhance(output, has_aligned=False, only_center_face=False, paste_back=True)
    cv2.imwrite(output_path, output)
    return True


def photo_handler(update: Update, context: CallbackContext):
    global upsampler, face_enhancer
    if upsampler is None or face_enhancer is None:
        upsampler, face_enhancer = init_models()

    photo_file = update.message.photo[-1].get_file()
    with tempfile.NamedTemporaryFile(suffix='.jpg') as tf_input, tempfile.NamedTemporaryFile(suffix='.jpg') as tf_output:
        photo_file.download(tf_input.name)
        success = enhance_image(tf_input.name, tf_output.name)
        if not success:
            update.message.reply_text("Не удалось обработать изображение.")
            return
        update.message.reply_photo(photo=open(tf_output.name, 'rb'))


def main():
    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, photo_handler))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()