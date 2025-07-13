import sys
import logging


class Logger:

    def __init__(self, filename, level=logging.INFO):
        logger = logging.getLogger()
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s %(message)s')

        file_handler = logging.FileHandler(filename, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)

        self.handler = (file_handler, stream_handler)

    def add_handler(self):
        for handler in self.handler:
            logger = logging.getLogger()
            logger.addHandler(handler)

    def remove_handler(self):
        for handler in self.handler:
            logger = logging.getLogger()
            logger.removeHandler(handler)
