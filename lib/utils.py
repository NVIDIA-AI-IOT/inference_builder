import os
import re
import sys
import logging
import time
from typing import Optional
import importlib
import numpy as np
import torch
import jinja2
from typing import List, Dict

kDebug = int(os.getenv("DEBUG", "0"))
PACKAGE_NAME = "NIM"

def stack_tensors_in_dict(list_of_tensor_dicts: List):
    """
    [{'k1': 'v1', 'k2': 'v2'}, {'k1': 'v3', 'k2': 'v4'}] ->
    {'k1': ['v1', 'v2'], 'k2': ['v3', 'v4']}
    """
    result = {}

    # Iterate over each dictionary in the list
    for d in list_of_tensor_dicts:
        for key, value in d.items():
            if key not in result:
                result[key] = []  # Create a new list for each new key
            result[key].append(value)  # Append the value to the list
    # Iterate over the combined dictionary
    for key in result:
        tensor_list = result[key]
        if isinstance(tensor_list[0], np.ndarray):
            result[key] = np.stack(tensor_list, axis=0)
        elif isinstance(tensor_list[0], torch.Tensor):
            result[key] = torch.stack(tensor_list, dim=0)

    return result

def split_tensor_in_dict(dict_of_tensor_list):
    """
    {'k1': ['v1', 'v2'], 'k2': ['v3', 'v4']} ->
    [{'k1': 'v1', 'k2': 'v2'}, {'k1': 'v3', 'k2': 'v4'}]
    """
    values = [dict_of_tensor_list[k] for k in dict_of_tensor_list]
    result = []
    length = min({len(v) for v in values})
    for i in range(length):
        result.append({ k: v[i] for k, v in dict_of_tensor_list.items()})

    return result

def import_class(module_name, class_name):
    # Import the module using importlib
    module = importlib.import_module(module_name)

    # Get the class from the module using getattr
    class_ = getattr(module, class_name)

    return class_

def create_jinja2_env():
    def start_with(field, s):
        return field.startswith(s)

    def replace(value, pattern, text):
        return re.sub(pattern, text, value)

    def extract(value, pattern):
        match = re.search(pattern, value)
        return match.group(1) if match else ''

    jinja2_env = jinja2.Environment()
    jinja2_env.tests["startswith"] = start_with
    jinja2_env.filters["replace"] = replace
    jinja2_env.filters["extract"] = extract
    jinja2_env.filters["zip"] = zip

    return jinja2_env

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