import os
from datetime import datetime

from paths import outputPath
from trainingDataLoader import loadTrainingData, matchPredictionOnInput
from trainModel import train
from preprocess_training_data import vectorizeData


DAYS_AHEAD = 10


def runTrainingCycle(inputFilePath):
    vectorPath = vectorizeData(inputFilePath)
    input, output = loadTrainingData(vectorPath)
    datasetX, datasetY = matchPredictionOnInput(input, output, lookback=64, daysAhead=10)

    encoderPath = "./output/models/encoder.h5"

    date = datetime.now()
    filename = os.path.basename(inputFilePath)
    filename = "{}_{}_{:%Y-%m-%d_%H%M%S}.csv".format(DAYS_AHEAD, filename, date)
    modelPath = "./output/models/model-" + filename + ".h5"

    predictionPath = outputPath + "predictions/predictions-" + filename + ".csv"

    if not os.path.exists(outputPath + "models/"):
        os.makedirs(outputPath + "models/")
    if not os.path.exists(outputPath + "predictions/"):
        os.makedirs(outputPath + "predictions/")
    train(filename, datasetX, datasetY, encoderPath, modelPath, predictionPath)


# TODO: just for debugging, remove in production
if __name__ == "__main__":
    runTrainingCycle("./data/autonomous-greenhouse_production_47_2020-06-05_165451.csv")
