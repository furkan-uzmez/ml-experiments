# logger.py
import logging
from datetime import datetime

def get_logger(log_path='training.log'):
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)

    # Her çağrıda yeni bir dosya handler'ı ekleyebilmek için mevcut handler'ları temizleyelim:
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    # Formatlayıcı
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y_%m_%d_%H_%M_%S')

    # Dosya yazıcı
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Konsola yazıcı
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
