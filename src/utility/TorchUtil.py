"""
Filename: TorchUtil.py
Description: This is a file that contains the class 'TorchUtil' to perform pytorch related utility function
"""

# Python Standard Libraries

# Third Party Libraries
import torch

# Project Module
from src.utility.FileUtil import FileUtil
from src.utility.LoggerUtil import LoggerUtil


# Source Code
class TorchUtil:
    """
    This class is responsible for pytorch related functionality/utility
    """

    def __init__(self):
        raise TypeError(f"class {self.__class__.__name__} is supposed to be a utility class, " +
                        "which should not be instantiated")

    @staticmethod
    def reloadModel(model, logger, path="", use_gpu=False):
        """
        Reload a previously trained model into the given 'model'
        ------------------------------------------------------------
        :param model:
        :param logger:
        :param path:
        :return:
        """
        logger.info("")
        logger.info("===================================")
        logger.info("** Trying to Load Previous Model **")
        logger.info("===================================")
        if path == "":
            logger.info("No trained model path specified. Start training from scratch")
            return model
        elif not FileUtil.fileExists(path):
            LoggerUtil.warning(f"Cannot find trained model at '{path}'. Start training from scratch", logger)
            return model
        else:
            model_dict = model.state_dict()
            if not use_gpu:
                pretrained_dict = torch.load(path, map_location=torch.device("cpu"))
            else:
                pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            logger.info(f"*** model at '{path}' has been successfully loaded! ***")
            return model

    @staticmethod
    def getLearningRates(optimizer):
        LR = []
        for param_group in optimizer.param_groups:
            LR.append(param_group["lr"])
        return LR







# For Testing Purposes
if __name__ == "__main__":
    pass