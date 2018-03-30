import os
from datetime import datetime

from paths import outputPath
from trainingDataLoader import loadTrainingData, matchPredictionOnInput
from trainModel import train
from Preprocess import vectorizeData


def runTrainingCycle(inputFilePath):
    vectorPath = vectorizeData(inputFilePath)
    input, output = loadTrainingData(vectorPath)
    datasetX, datasetY = matchPredictionOnInput(input, output, lookback=50, daysAhead=10)

    encoderPath = "./output/models/encoder.h5"

    date = datetime.now()
    daysAhead = 10
    filename = "{}_{}_{:%Y-%m-%d_%H:%M:%S}.csv".format(daysAhead, filename, date)
    modelPath = "./output/models/model-" + filename + ".h5"

    predictionPath = outputPath + "predictions/predictions-" + filename + ".csv"

    if not os.path.exists(outputPath + "models/"):
        os.makedirs(outputPath + "models/")
    if not os.path.exists(outputPath + "predictions/"):
        os.makedirs(outputPath + "predictions/")
    train(filename, datasetX, datasetY, encoderPath, modelPath, predictionPath)


# TODO: just for debugging, remove in production
if __name__ == "__main__":
    runTrainingCycle("./data/ABCPlants_production_10_2018-03-05_13:15:36.csv")
