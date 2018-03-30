import os, calendar
import numpy as np
import pandas as pd

from paths import preprocessPath


def dateTimeToVector(sheet):
    sheet = sheet.reset_index(drop=True)
    #sheet = sheet.fillna(0)
    columnnames = sheet.columns.values
    time = sheet[0]
    year = time[0].year

    daysInYear = 365
    if calendar.isleap(year):
        daysInYear = 366
    # ## Remove last rows (total amounts)
    index = sheet.index[daysInYear:]
    sheet = sheet.drop(index)

    # ## Description
    cleanedSheet = sheet[requiredNames]
    cleanedSheet = cleanedSheet.apply(pd.to_numeric)

    print(cleanedSheet.describe())

    sheetDates = sheet['DATE']
    # Create cyclic date parameter
    dateX = []
    dateY = []
    for i in range(daysInYear):
        x = np.sin(i / daysInYear * np.pi * 2)
        y = np.cos(i / daysInYear * np.pi * 2)
        # Prevent 0 values
        if x == 0:
            x = .000001
        if y == 0:
            y = .000001
        dateX.append(x)
        dateY.append(y)

    dateCycle = pd.DataFrame({'dateX': dateX, 'dateY': dateY})

    cleanedSheet = pd.concat([sheetDates, dateCycle, cleanedSheet], axis=1)

    return cleanedSheet


def categoryToVector(data):
    #TODO: Write
    pass



def vectorizeData(filepath):
    filename = os.path.basename(filepath)
    meansPath = preprocessPath + filename + "_means.npy"
    stddevPath = preprocessPath + filename + "_stddev.npy"
    vectorPath = preprocessPath + filename + "_vector.npy"
    if os.path.isfile(meansPath) and os.path.isfile(stddevPath) and os.path.isfile(vectorPath):
        print("Skipping vectorization, already done")
        return vectorPath

    data = pd.read_csv(filepath, sep=';', decimal='.')
    data = dateTimeToVector(data)
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

    np.save(meansPath, means)
    np.save(stddevPath, stddev)
    np.save(vectorPath, vector)

    return vectorPath
