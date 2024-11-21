class LogsManager:
    def __init__(self):
        import loguru

        self._logger = loguru.logger

    def connect_log_file(self):
        self._logger.add("../logs/{time}.log", rotation="00:00", encoding="utf-8", enqueue=True, level="DEBUG")

    @property
    def logger(self):
        return self._logger
