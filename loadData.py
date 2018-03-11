import time
import os
import numpy as np
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from paths import preprocessPath, vectorPath
import pandas as pd


class DataHandler(PatternMatchingEventHandler):
    patterns = ["*.csv"]

    def process(self, event):
        """
        event.event_type
            'modified' | 'created' | 'moved' | 'deleted'
        event.is_directory
            True | False
        event.src_path
            path/to/observed/file
        """
        data = pd.read_csv(event.src_path, sep=',', decimal='.')
        vectorizeData(data)

    def on_modified(self, event):
        self.process(event)

    def on_created(self, event):
        self.process(event)


def vectorizeData(data):
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

    np.save(vectorPath, vector)


if __name__ == '__main__':
    dataPath = os.path.abspath("./data")
    observer = Observer()
    observer.schedule(DataHandler(), path=dataPath)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
