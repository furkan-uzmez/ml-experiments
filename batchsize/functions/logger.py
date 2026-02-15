# logger.py
import logging
from datetime import datetime

def get_logger(log_path='training.log'):
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)

    # Aynı logger birden fazla handler eklenmesini önlemek için:
    if not logger.handlers:
        # Formatlayıcı
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y_%m_%d_%H_%M_%S')

        # Dosya yazıcı
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Konsola yazıcı
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
