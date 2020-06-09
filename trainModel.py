import os
import csv

import numpy as np
from keras.callbacks import TensorBoard
from keras.models import load_model

from paths import preprocessPath

from AutoEncoder import AutoEncoder
from models import *

NUMBER_OF_PREDICTED_DAYS = 1
TRAINING_DATA_PERCENTAGE = 0.8
LOOKBACK = 64
DAYS_AHEAD = 1
USE_AUTOENCODER = False


def train(filename, datasetX, datasetY, encoderPath, modelPath, predictionPath):
    # Split into training and testing set.
    index = int(len(datasetX) * TRAINING_DATA_PERCENTAGE)
    trainingX = np.array(datasetX[:index])
    trainingY = np.array(datasetY[:index])
    testingX = np.array(datasetX[index:])
    testingY = np.array(datasetY[index:])

    # Remove timestamps from training.
    trainingY = trainingY.transpose()[1].transpose()
    # Extract timestamps from testing.
    testTargetDates = testingY.transpose()[0].transpose()
    testingY = testingY.transpose()[1].transpose()

    numberOfInputParameters = len(datasetX[0][0])
    inputShape = (LOOKBACK, numberOfInputParameters)

    if USE_AUTOENCODER:
        if os.path.isfile(encoderPath):
            encoder = load_model(encoderPath)
        else:
            aec = AutoEncoder(inputShape)
            encoder = aec.fit(trainingX, testingX)
            encoder.save(encoderPath)

        for layer in encoder.layers:
            layer.trainable = False

    if os.path.isfile(modelPath):
        model = load_model(modelPath)
    else:
        outputShape = (1)
        if USE_AUTOENCODER:
            model = createModelWithEncoder(encoder, outputShape)
        else:
            model = createConvModel(inputShape, outputShape)

        # Fit the model.
        model.fit(trainingX, trainingY,
                  epochs=150,
                  batch_size=256,
                  validation_data=[testingX, testingY],
                  callbacks=[TensorBoard(log_dir='/tmp/')])
        model.save(modelPath)

    # Evaluate the model.
    last = len(trainingX[0][0]) - 1
    mean = np.load(preprocessPath + filename + "_means.npy")[last]
    stddev = np.load(preprocessPath + filename + "_stddev.npy")[last]

    prediction = model.predict(testingX)

    with open(predictionPath, 'w') as csvfile:
        trainScores = model.evaluate(trainingX, trainingY)
        testScores = model.evaluate(testingX, testingY)
        predictionWriter = csv.writer(csvfile, delimiter=';',
                                      quotechar='|', quoting=csv.QUOTE_MINIMAL)
        predictionWriter.writerow(['Test Loss', str(testScores), '', 'Training Loss', str(trainScores)])
        predictionWriter.writerow(['Date', 'Prediction', 'Target', 'Prediction/Target'])
        lines = []
        for prediction, target, date in zip(prediction, testingY, testTargetDates):
            prediction = int(prediction[0] * stddev + mean)
            target = int(target * stddev + mean)
            fraction = prediction / target
            lines += [[str(date.date()), str(prediction), str(target), str(fraction)]]

        lines = sorted(lines)

        for line in lines:
            predictionWriter.writerow(line)

        print(testScores)


if __name__ == "__main__":
    pass
    # for i in range(1, 61):
    #     DAYS_AHEAD = i
    #     encoderPath = "./output/models/encoder.h5"
    #
    #     date = datetime.now()
    #     filename = "{}_{}_{}_{:%Y-%m-%d_%H%M%S}.csv".format(clientName, outputType, inputSize, date)
    #     modelPath = "./output/models/model-" + str(DAYS_AHEAD) + filename + ".h5"
    #
    #     predictionPath = outputPath + "predictions/predictions-" + str(DAYS_AHEAD) + ".csv"
    #     makeDirIfNotExists(encoderPath)
    #     makeDirIfNotExists(modelPath)
    #     makeDirIfNotExists(predictionPath)
    #     train(encoderPath, modelPath, predictionPath)
