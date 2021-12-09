"""
Filename: AudioUtil.py
Description: This is a file that contains the class ConfigUtil for reading/writing to config file
"""

# Python Standard Libraries
import argparse
import json
from jsonschema import validate

# Third Party Libraries

# Project Module
from src.utility.FileUtil import FileUtil


# Source Code
class ConfigType:
    """
    This class is just holding the json datatype in config file as global constants for the 'ConfigUtil' class
    """

    INTEGER = {"type": "integer"}
    NUMBER = {"type": "number"}
    BOOLEAN = {"type": "boolean"}
    STRING = {"type": "string"}

    def __init__(self):
        raise TypeError(f"class {self.__class__.__name__} is supposed to be a utility class, " +
                        "which should not be instantiated")

class ConfigUtil:
    """
    This class is responsible for handling config file
    """

    def __init__(self):
        raise TypeError(f"class {self.__class__.__name__} is supposed to be a utility class, " +
                        "which should not be instantiated")

    @staticmethod
    def readConfig(configFilePath) -> dict:
        """
        Try to read the JSON configuration file. If failed, throw exception.
        ------------------------------------------------------------
        :param configFilePath:
        :return: a dictionary of key-value pair
        """
        # Check if the config file exists
        if not FileUtil.fileExists(configFilePath):
            raise ValueError(f"Cannot find the config file at '{FileUtil.resolvePath(configFilePath)}'")

        # Try to read the config file. If failed, throw exception.
        try:
            with open(configFilePath, 'r') as configFile:
                config = json.load(configFile)
        except Exception as e:
            raise RuntimeError(f"Failed to read the config file at '{FileUtil.resolvePath(configFilePath)}' due to exception '{e}'")
        else:
            return config

    @staticmethod
    def readAudioTrainConfig(configFilePath) -> dict:
        """
        Read the [Audio] section of the train_config file located at 'configFilePath'
        ------------------------------------------------------------
        :param configFilePath:
        :return: a dictionary of key-value pair
        """
        def checkAudioConfigFormat(audioConfig):
            schema = {
                "type": "object",
                "properties" : {
                    "nClasses":             ConfigType.INTEGER,
                    "model-path":           ConfigType.STRING,
                    "sorted-labels-path":   ConfigType.STRING,
                    "dataset-path":         ConfigType.STRING,
                    "mode":                 ConfigType.STRING,
                    "every-frame":          ConfigType.BOOLEAN,
                    "lr":                   ConfigType.NUMBER,
                    "batch-size":           ConfigType.INTEGER,
                    "workers":              ConfigType.INTEGER,
                    "epochs":               ConfigType.INTEGER,
                    "print-stat-interval":  ConfigType.INTEGER,
                    "test":                 ConfigType.BOOLEAN
                }
            }
            validate(instance=audioConfig, schema=schema)

        # read the config
        config = ConfigUtil.readConfig(configFilePath)
        audioConfig = config['Audio']

        # check format of the config
        try:
            checkAudioConfigFormat(audioConfig)
        except Exception as e:
            raise RuntimeError(f"Incorrect format in the config file '{FileUtil.resolvePath(configFilePath)}': {e}")

        return audioConfig

    @staticmethod
    def readVideoTrainConfig(configFilePath) -> dict:
        """
        Read the [Video] section of the train_config file located at 'configFilePath'
        ------------------------------------------------------------
        :param configFilePath:
        :return: a dictionary of key-value pair
        """
        def checkVideoConfigFormat(videoConfig):
            schema = {
                "type": "object",
                "properties": {
                    "nClasses": ConfigType.INTEGER,
                    "model-path": ConfigType.STRING,
                    "sorted-labels-path": ConfigType.STRING,
                    "dataset-path": ConfigType.STRING,
                    "mode": ConfigType.STRING,
                    "every-frame": ConfigType.BOOLEAN,
                    "lr": ConfigType.NUMBER,
                    "batch-size": ConfigType.INTEGER,
                    "workers": ConfigType.INTEGER,
                    "epochs": ConfigType.INTEGER,
                    "print-stat-interval": ConfigType.INTEGER,
                    "test": ConfigType.BOOLEAN
                }
            }
            validate(instance=videoConfig, schema=schema)

        # read the config
        config = ConfigUtil.readConfig(configFilePath)
        videoConfig = config['Video']

        # check format of the config
        try:
            checkVideoConfigFormat(videoConfig)
        except Exception as e:
            raise RuntimeError(f"Incorrect format in the config file '{FileUtil.resolvePath(configFilePath)}': {e}")

        return videoConfig

    @staticmethod
    def readAudioVisualTrainConfig(configFilePath) -> dict:
        """
        Read the [AudioVisual] section of the train_config file located at 'configFilePath'
        ------------------------------------------------------------
        :param configFilePath:
        :return: a dictionary of key-value pair
        """

        def checkAudioVisualConfigFormat(audioVisualConfig):
            schema = {
                "type": "object",
                "properties": {
                    "nClasses": ConfigType.INTEGER,
                    "audio-path": ConfigType.STRING,
                    "video-path": ConfigType.STRING,
                    "concat-path": ConfigType.STRING,
                    "audio-dataset": ConfigType.STRING,
                    "video-dataset": ConfigType.STRING,
                    "sorted-labels-path": ConfigType.STRING,
                    "mode": ConfigType.STRING,
                    "every-frame": ConfigType.BOOLEAN,
                    "lr": ConfigType.NUMBER,
                    "batch-size": ConfigType.INTEGER,
                    "workers": ConfigType.INTEGER,
                    "epochs": ConfigType.INTEGER,
                    "print-stat-interval": ConfigType.INTEGER,
                    "test": ConfigType.BOOLEAN
                }
            }
            validate(instance=audioVisualConfig, schema=schema)

        # read the config
        config = ConfigUtil.readConfig(configFilePath)
        audioVisualConfig = config['AudioVisual']

        # check format of the config
        try:
            checkAudioVisualConfigFormat(audioVisualConfig)
        except Exception as e:
            raise RuntimeError(f"Incorrect format in the config file '{FileUtil.resolvePath(configFilePath)}': {e}")

        return audioVisualConfig

    # @staticmethod
    # def getConfigPath():
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--config', default='./train_config.json', help="path to 'train_config.json' file")
    #     args = parser.parse_args()
    #     return args.config




# For Testing Purposes
if __name__ == "__main__":
    config = ConfigUtil.readCommonConfig(r"S:\College\UCB\2021 Fall\EE225D\Projects\AudioVisualProj\src\config.json")
    print(config)