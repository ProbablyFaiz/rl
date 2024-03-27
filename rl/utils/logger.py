import logging


class LogFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    white = "\x1b[37;0m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    datefmt = "%Y-%m-%d %H:%M:%S"
    format = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: white + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(LogFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
LOGGER = get_logger(__name__)
