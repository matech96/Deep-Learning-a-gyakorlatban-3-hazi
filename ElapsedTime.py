import time


class ElapsedTime:
    def __init__(self, process_name: str="Process", is_ms: bool=False):
        """
        This class should be used in with statement. It measures the time spend in the statement.
        :param process_name: Name of the process. Used in the displayed string.
        :param is_ms: If true the output is in milliseconds, seconds otherwise.
        """
        self.is_ms = is_ms
        self.process_name: str = process_name
        self.start: float = -1
        self.end: float = -1

    def __enter__(self):
        self.start: float = time.time()

    def __exit__(self, ignored, ignored1, ignored2):
        self.end: float = time.time()
        if self.is_ms:
            elapsed_time_ms: float = (self.end - self.start) * 1000
            print("%s was running for %f ms" % (self.process_name, elapsed_time_ms))
        else:
            elapsed_time: float = (self.end - self.start)
            print("%s was running for %f s" % (self.process_name, elapsed_time))
