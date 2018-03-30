import os
from runTrainingCycle import runTrainingCycle


def getFilePath():
    files = os.listdir("data/")
    file = files[0]
    print("File found: " + file)
    return "data/" + file


filepath = getFilePath()
runTrainingCycle(filepath)
