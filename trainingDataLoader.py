import numpy as np


def loadTrainingData(vectorPath):
    allData = np.load(vectorPath)

    # Remove dates from input
    input = allData.transpose()[1:].transpose()

    # Extract training target date and value
    last = len(allData[0]) - 1
    output = np.array([[vector[0], vector[last]] for vector in allData])
    return input, output


def matchPredictionOnInput(inputDataset, predictionDataset, lookback, daysAhead):
    """Take input and prediction arrays and lookback amount.
    Create two arrays of *lookback* input values and predictions,
    where matching indices correspond."""
    if len(inputDataset) < lookback - (daysAhead - 1):
        raise ValueError(
            "Lookback of {} is impossible, dataset contains only {} items".format(lookback, len(inputDataset)))

    datasetY = predictionDataset[lookback + (daysAhead - 1):]

    datasetX = []
    for i in range(len(datasetY)):
        inputValues = [inputDataset[i + j] for j in range(lookback)]
        datasetX.append(inputValues)

    return datasetX, datasetY
