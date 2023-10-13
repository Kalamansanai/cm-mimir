import logging

LOG_PATH = "../assets/logger.log"

_logger = logging.Logger("cm-logger", level=logging.DEBUG)
_fh = logging.FileHandler(LOG_PATH)
_fh.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
_fh.setLevel(logging.DEBUG)
_logger.addHandler(_fh)


class Logger:
    def debug(self, msg: str):
        _logger.debug(msg)

    def info(self, msg: str):
        _logger.info(msg)

    def warning(self, msg: str):
        _logger.warning(msg)

    def error(self, msg: str):
        _logger.error(msg)


logger = Logger()
