import os
from runTrainingCycle import runTrainingCycle


def getFilePaths():
    files = os.listdir("data/")
    print("File(s) found: " + str(files))
    files = ["data/" + file for file in files]
    return files


if __name__ == "__main__":
    filepaths = getFilePaths()
    for filepath in filepaths:
        runTrainingCycle(filepath)
