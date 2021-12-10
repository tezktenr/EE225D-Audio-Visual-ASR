"""
Filename: LoggerUtil.py
Description: This is a file that contains the class 'LoggerUtil' for logging related utility function
"""

# Python Standard Libraries
import logging
import warnings
from datetime import datetime
import time

# Third Party Libraries

# Project Module


# Source Code
class LoggerUtil:
    """
    This class is responsible for logging related utility
    """

    def __init__(self):
        raise TypeError(f"class {self.__class__.__name__} is supposed to be a utility class, " +
                        "which should not be instantiated")

    @staticmethod
    def getLogger(loggerName, logFilePath=None):
        logger = logging.getLogger(loggerName)
        logger.setLevel(logging.INFO)

        # setup basic logging format
        logFormatter = logging.Formatter(fmt='[%(levelname)s] %(message)s')

        # setup logging to the LOG FILE
        if logFilePath:
            fh = logging.FileHandler(logFilePath, mode='a')
            fh.setLevel(logging.INFO)
            fh.setFormatter(logFormatter)
            logger.addHandler(fh)

        # setup logging to the CONSOLE
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logFormatter)
        logger.addHandler(console)

        return logger

    @staticmethod
    def logCurrentTime(logger):
        now = datetime.now()
        logger.info(now)

    @staticmethod
    def warning(msg, logger=None):
        warnings.formatwarning = lambda msg, *args, **kwargs: str(msg) + '\n'
        warnings.warn(f"[Warning] {msg}")

        # Unfinished feature
        # -----------------------
        # if (logger is not None):
        #     logger.warning(msg)
        # -----------------------

    @staticmethod
    def printAllStat(logger, begin_time, batch_idx, running_loss, running_corrects, running_all, data_loader):
        logger.info('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tAcc:{:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                running_all,
                len(data_loader.dataset),
                100. * batch_idx / (len(data_loader) - 1),
                running_loss / running_all,
                running_corrects / running_all,
                time.time() - begin_time,
                (time.time() - begin_time) * (len(data_loader) - 1) / batch_idx - (time.time() - begin_time)  )
        )





# For Testing Purposes
if __name__ == "__main__":
    pass