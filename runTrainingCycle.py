import numpy as np
import os
import pandas as pd
import ntpath

from paths import preprocessPath
from trainingDataLoader import loadTrainingData, matchPredictionOnInput


def runTrainingCycle(filepath):
    data = pd.read_csv(filepath, sep=',', decimal='.')
    filename = ntpath.basename(filepath).split("_")[:3]
    filename = "_".join(filename)
    vectorizeData(filename, data)
    vectorPath = preprocessPath + filename + "_vector.npy"
    input, output = loadTrainingData(vectorPath)
    datasetX, datasetY = matchPredictionOnInput(input, output, lookback=50, daysAhead=10)

def vectorizeData(filename, data):
    vector = np.array(data)
    # Ignore date while calculating means/stddevs
    vector = np.transpose(vector)
    dates = vector[0]
    vector = vector[1:]

    vector = vector.transpose()
    means = np.mean(vector, axis=0)
    vector = np.subtract(vector, means)
    stddev = np.std(vector, dtype=float, axis=0)
    vector = np.divide(vector, stddev)
    vector = vector.transpose()

    vector = np.vstack([dates, vector])
    vector = np.transpose(vector)

    if not os.path.exists(preprocessPath):
        os.makedirs(preprocessPath)

    vectorPath = preprocessPath + filename + "_vector.npy"
    np.save(vectorPath, vector)


# TODO: just for debugging, remove in production
if __name__ == "__main__":
    runTrainingCycle("./data/ABCPlants_production_10_2018-03-05_13:15:36.csv")
