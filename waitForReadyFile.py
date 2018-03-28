import time
import os
from runTrainingCycle import runTrainingCycle


def readyFileHasAppeared():
    files = os.listdir("data/")
    for file in files:
        if file == "ready":
            return True
    return False


def getFilePath():
    files = os.listdir("data/")
    for file in files:
        if file != "ready":
            print("File found: " + file)
            return "data/" + file
    raise Exception("No trainable file found, even though multiple files are present.")


while True:
    if readyFileHasAppeared():
        filepath = getFilePath()
        runTrainingCycle(filepath)
    else:
        time.sleep(1)
