import os
import sys
import logging
import time
from typing import Optional

kDebug = int(os.getenv("DEBUG", "0"))
PACKAGE_NAME = "NIM"

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """ Get component logger

        Parameters:
        name: a module name

        Returns: A Logger Instance
    """

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    if kDebug:
        log_level = logging.DEBUG
    name = f"{PACKAGE_NAME}.{name}"
    log_format = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
    # sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    print(f"logger {str(name)}, log_level: {str(log_level)}")
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    logger = logging.getLogger(name)
    logger.propagate = True
    # formatter = logging.Formatter(log_format)
    # stream_handler = logging.StreamHandler(sys.stdout)
    # stream_handler.setFormatter(formatter)
    # stream_handler.setLevel(log_level)
    # logger.handlers.clear()
    # logger.addHandler(stream_handler)
    return logger

def flush(logger):
    for h in logger.handlers:
        h.flush()


class SimpleLogger:
    def __init__(self, name: str = ""):
        self.name = name
    def log_print(self, level, *args, **kwargs):
        asctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        prefix = f"{asctime} [{str(level)}] [{self.name}]: "
        print(prefix, *args, **kwargs)

    def debug(self, *args, **kwargs):
        if kDebug:
            self.log_print("DEBUG", *args, **kwargs)

    def info(self, *args, **kwargs):
        self.log_print("INFO", *args, **kwargs)

    def warning(self, *args, **kwargs):
        self.log_print("WARNING", *args, **kwargs)

    def error(self, *args, **kwargs):
        self.log_print("ERROR", *args, **kwargs)

def tensor_info(tensor):
    return f" shape: {tensor.shape}, dtype: {tensor.dtype}, dtype: {tensor.device}"