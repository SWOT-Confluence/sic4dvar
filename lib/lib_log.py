import logging
import os

def set_logger(loglevel, filename=None):

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-4.4s]  %(message)s")
    logger = logging.getLogger()
    logger.setLevel(level=loglevel)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    
    #if not filename.parent.exists():
    #   filename.parent.mkdir(parents=True, exist_ok=True)
    
    if filename:
        if os.path.exists(filename):
            os.remove(filename)
        fileHandler = logging.FileHandler(filename)
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)
