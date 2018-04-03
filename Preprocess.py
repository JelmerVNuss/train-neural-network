import os, calendar
import numpy as np
import pandas as pd
from datetime import datetime

from paths import preprocessPath


# Converta datetime to a year-cyclic date parameter
def dayToCycle(date):
    startOfYear = datetime(date.year, 1, 1)
    startOfNextYear = datetime(date.year + 1, 1, 1)
    difference = startOfNextYear - startOfYear
    daysInYear = difference.days

    daysSinceStart = (date - startOfYear).days
    cycleIndex = daysSinceStart / daysInYear * np.pi * 2

    x = np.sin(cycleIndex)
    y = np.cos(cycleIndex)
    # Prevent 0 values
    if x == 0:
        x = .000001
    if y == 0:
        y = .000001
    return x, y


def dateTimeToVector(sheet):
    sheet = sheet.reset_index(drop=True)
    #sheet = sheet.fillna(0)
    columnnames = list(sheet.columns.values)
    dateSlice = sheet['Date']

    dates = [datetime.strptime(date, "%m/%d/%y") for date in dateSlice]

    cycles = [dayToCycle(date) for date in dates]
    dateX = [cycle[0] for cycle in cycles]
    dateY = [cycle[1] for cycle in cycles]

    columnnames.remove('Date')
    cleanedSheet = sheet[columnnames]
    #cleanedSheet = cleanedSheet.apply(pd.to_numeric)

    print(cleanedSheet.describe())

    sheetDates = sheet['Date']
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
