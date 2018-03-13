import time
import os
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from runTrainingCycle import runTrainingCycle


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
        runTrainingCycle(event.src_path)

    def on_modified(self, event):
        self.process(event)

    def on_created(self, event):
        self.process(event)


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
