"""
Filename: FileUtil.py
Description: This is a file that contains the class FileUtil for file and filesystem related utility
Warning:
    * pathlib is not supported by python2
"""

# Python Standard Libraries
import platform
from pathlib import Path

# Third Party Libraries

# Project Module


# Source Code
class FileUtil:
    """
    This class is responsible for general file/filesystem utility
    """

    # Supported/Tested Platforms
    supportedPlatforms = ["Windows"]

    def __init__(self):
        raise TypeError(f"class {self.__class__.__name__} is supposed to be a utility class, " +
                        "which should not be instantiated")

    @staticmethod
    def checkFileExists(paths: list) -> [str]:
        """
        This method takes in a LIST/TUPLE of path names and returns those path names that don't exist in the filesystem
        ------------------------------------------------------------
        :param paths: a list/tuple of pathname
        :return: a list containing those pathnames that doesn't exist in the filesystem
        """
        if (type(paths) is not list and type(paths) is not tuple):
            raise TypeError("The argument 'paths' must be either a list or a tuple")
        nonExistentFiles = [path for path in paths if not Path(path).exists()]
        return nonExistentFiles


    @staticmethod
    def getFileExtension(pathname):
        """
        This method extracts the file extension of the file located in the path name as specified

        Note:
            * will throw exception if file doesn't exist
            * will throw exception if current operating system is not supported
            * will throw exception if the file has no file extension
        ------------------------------------------------------------
        :param pathname: the pathname for a file
        :return: file extension (no 'dot' in the front) - e.g. 'wav', 'mp3'
        """
        # Check if the current operating system is supported/compatible
        if (platform.system() not in FileUtil.supportedPlatforms):
            raise ValueError(f"The operating system '{platform.system()}' is not supported " +
                             f"for the method '{FileUtil.getFileExtension.__name__}'")

        filePath = Path(pathname)

        # Make sure file exists in the file system
        if (not filePath.exists()):
            raise ValueError(f"The file '{pathname}' doesn't exist in the file system")

        # Get file extension
        fileExtension = filePath.suffix

        # Make sure file has a file extension
        if (len(fileExtension) <= 0):
            raise ValueError(f"The file '{pathname}' has no file extension")

        # (On Windows OS) File extension should start with a dot
        if (not fileExtension.startswith('.')):
            raise ValueError(f"The file extension '{fileExtension}' doesn't start with '.'")

        return fileExtension[1:]





# For Testing Purposes
if __name__ == "__main__":
    FileUtil.checkFileExists(["file1", "file2"])