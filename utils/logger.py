import logging

class Logger:
    def __init__(self):
        logging.basicConfig(format='%(asctime)s %(levelname)-7s : %(message)s', level=logging.INFO)

    def info(self, msg):
        logging.info(msg)
    
    def warning(self, msg):
        logging.warning(msg)
    
    def error(self, msg):
        logging.error(msg)