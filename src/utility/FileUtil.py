"""
Filename: FileUtil.py
Description: This is a file that contains the class FileUtil for file and filesystem related utility
Warning:
    * pathlib is not supported by python2
"""

# Python Standard Libraries
import os
import shutil
import platform
from pathlib import Path


# Third Party Libraries

# Project Module
from src.utility.LoggerUtil import LoggerUtil

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
    def fileExists(filepath: str) -> bool:
        """
        Check if the given file path exists in the file system AND whether it is a regular file
        ------------------------------------------------------------
        :param filepath:
        :return: boolean
        """
        p = Path(str(filepath))
        return p.exists() and p.is_file()

    @staticmethod
    def directoryExists(dirpath: str) -> bool:
        """
        Check if the given directory path exists in the file system AND whether it is a directory
        ------------------------------------------------------------
        :param dirpath:
        :return: boolean
        """
        p = Path(str(dirpath))
        return p.exists() and p.is_dir()

    @staticmethod
    def checkFilesExist(filepaths: list) -> [str]:
        """
        This method takes in a LIST/TUPLE of file path names and
        returns those path names that don't exist in the filesystem
        ------------------------------------------------------------
        :param paths: a list/tuple of pathname
        :return: a list containing those pathnames that doesn't exist in the filesystem
        """
        if (type(filepaths) is not list and type(filepaths) is not tuple):
            raise TypeError("The argument 'paths' must be either a list or a tuple")

        nonExistentFiles = []
        for filepath in filepaths:
            path = Path(str(filepath))
            if (not path.exists() or not path.is_file()):
                nonExistentFiles.append(str(path))
        return nonExistentFiles

    @staticmethod
    def makeDirRecursively(path: str):
        """
        create all the (including parent) directory in the file system as specified in path

        Warning: the last section of the path must also be specified as a directory and not a file
            =======================================================
            ** Here is an Example: makeDirRecursively("/tmp/a/b/c")
            =======================================================
            **  tmp/
            **  ├─ a/
            **  │  ├─ b/
            **  │  │  ├─ c/
            **  │  │  │  ├─ new_file.txt
            **
            =======================================================
        ------------------------------------------------------------
        :param path: the path (type: str)
        :return:
        """
        p = Path(str(path))
        p.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def removeDirRecursively(path: str, raiseErrorIfNotDeleted = False, forceDelete = False):
        """
        Delete a possibly non-empty directory and all the files inside the directory

        Note: path must point to a directory and not a file
        ------------------------------------------------------------
        :param path:
        :param raiseErrorIfNotDeleted: if true, this function will raise an exception if the deletion failed
        :param forceDelete: if true, the function will not prompt for user's confirmation for deleting the directory
        :return:
        """
        if (not FileUtil.directoryExists(path)):
            if (raiseErrorIfNotDeleted):
                raise ValueError(f"The path '{path}' is not a directory or doesn't exist and would not be deleted")
            else:
                LoggerUtil.warning(f"The path '{path}' is not a directory or doesn't exist and would not be deleted")
                return

        user_want_to_delete = False
        if (not forceDelete):
            user_input = input(f"Are you sure you want to clear the directory '{path}' (Y/[N]):").strip().upper()
            user_want_to_delete = user_input in ['Y', 'YES']    # if user enter invalid input, it will be assumed as a 'NO'

        if (forceDelete or user_want_to_delete):
            shutil.rmtree(Path(str(path)))
        else:
            if (raiseErrorIfNotDeleted):
                raise ValueError(f"User rejected to delete the path '{path}'")
            else:
                LoggerUtil.warning(f"User rejected to delete the path '{path}'")
                return

    @staticmethod
    def removeFile(path: str, raiseErrorIfNotDeleted = False, forceDelete = False):
        """
        remove the file as specified by 'path'.

        Note: the path must point to a file and not a directory
        ------------------------------------------------------------
        :param path:
        :param raiseErrorIfNotDeleted: if true, this function will raise an exception if the deletion failed
        :param forceDelete: if true, the function will not prompt for user's confirmation for deleting the file
        :return:
        """
        p = Path(str(path))
        if (not p.exists() or not p.is_file()):
            if (raiseErrorIfNotDeleted):
                raise ValueError(f"The path '{p}' is not a file or doesn't exist and would not be deleted")
            else:
                LoggerUtil.warning(f"The path '{p}' is not a file or doesn't exist and would not be deleted")
                return

        user_want_to_delete = False
        if (not forceDelete):
            user_input = input(f"Are you sure you want to delete the file '{p}' (Y/[N]):").strip().upper()
            user_want_to_delete = user_input in ['Y', 'YES']  # if user enter invalid input, it will be assumed as a 'NO'

        if (forceDelete or user_want_to_delete):
            p.unlink()
        else:
            if (raiseErrorIfNotDeleted):
                raise ValueError(f"User rejected to delete the path '{p}'")
            else:
                LoggerUtil.warning(f"User rejected to delete the path '{p}'")
                return

    @staticmethod
    def getFileExtension(pathname: str) -> str:
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

        filePath = Path(str(pathname))

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


    @staticmethod
    def joinPath(*pathArgs) -> str:
        """
        return the path that are formed by joining all the paths in *pathArgs together
        ------------------------------------------------------------
        :param pathArgs: multiple paths (variable length argument list)
        :return: joined path (type: str)
        """
        if (len(pathArgs) <= 0):
            return ""
        joinedPath = Path(str(pathArgs[0]))
        for i in range(1,len(pathArgs)):
            joinedPath = joinedPath / pathArgs[i].strip("/\\")
        return str(joinedPath)

    @staticmethod
    def extractPartsFromPaths(pathname: str) -> tuple:
        """
        Extract different sections of the path
        Note: this function is intended to be used as OS-independent,
              thereby will handle both back-slash and forward-slash
            =======================================================
            ** Here is an Example:
            =======================================================
            **  parts = extractPartsFromFile(r'C:/folder1//folder2\folder3\\folder4/file')
            **  print(parts)
            **  >>> parts = ('C:\\', 'folder1', 'folder2', 'folder3', 'folder4', 'file')
            =======================================================
        ------------------------------------------------------------
        :param pathname: str
        :return: tuple of strs
        """
        path = Path(str(pathname))
        return path.parts

    @staticmethod
    def resolvePath(path: str) -> str:
        """
        Resolve the path given
            =======================================================
            ** Here is an Example:
            =======================================================
            **  ## Suppose our working directory currently is "/folder1/folder2/folder3"
            **  realPath = resolvePath("../../file")
            **  print(realPath)
            **  >>> realPath = "/folder1/file"
            =======================================================
        ------------------------------------------------------------
        :param path:
        :return:
        """
        return str(Path(str(path)).resolve())

    @staticmethod
    def getDirectoryOfFile(filepath: str) -> str:
        """
        Get the parent directory of a given path
            =======================================================
            ** Here is an Example:
            =======================================================
            **  filepath = "/tmp/dir1/file.txt"
            **  parentPath = getDirectoryOfFile(filepath)
            **  print(parentPath)
            **  >>> parentPath = "/tmp/dir1"
            =======================================================
        ------------------------------------------------------------
        :param filepath:
        :return:
        """
        p = Path(str(filepath))
        return str(p.parents[0])




# For Testing Purposes
if __name__ == "__main__":
    print(FileUtil.resolvePath("../hi"))