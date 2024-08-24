
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from typing import List
from abc import ABC, abstractmethod
from typing import Callable
from common.config import config
from common.utils import get_logger
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from dataclasses import dataclass

logger = get_logger(__name__)

@dataclass
class Error:
    message: str

    def __bool__(self):
        return False

class DataFlow:
    """A single input flow to a model"""
    def __init__(self, names: List, timeout=10):
        self._names = names
        self._queue = Queue()
        self._timeout = timeout

    @property
    def names(self):
        return self._names

    def put(self, item):
        self._queue.put(item, timeout=self._timeout)

    def get(self):
        try:
            item = self._queue.get(timeout=self._timeout)
        except Empty:
            item = Error("timeout")
        return item


class ModelBackend(ABC):
    """Interface for standardizing the model backend """
    @abstractmethod
    def __call__(self, **kwargs):
        raise Exception("Not Implemented")

class ModelOperator:
    """An model operator runs a single model"""
    def __init__(self, model_config, check_point_dir):
        self._model_name = model_config.name
        self._check_point_dir = check_point_dir
        self._in = []
        self._out = [DataFlow([o.name for o in model_config.output])]
        self._running = False
        self._tokenizer = None
        if hasattr(model_config, "tokenizer"):
            if model_config.tokenizer.type == "auto":
                self._tokenizer = AutoTokenizer.from_pretrained(
                    f"{self._check_point_dir}/{self._model_name}", use_fast=True, use_legacy=False
                )
                self._tokenizer.input = model_config.tokenizer.input
                self._tokenizer.output = model_config.tokenizer.output

    @property
    def model_name(self):
        return self._model_name

    @property
    def outputs(self):
        return self._out.copy()

    def inject(self, input_names: List[str]):
        input = DataFlow(input_names)
        self._in.append(input)
        logger.debug(f"Input {input_names} connected to model {self._model_name}")
        return input


    def run(self, model_backend: ModelBackend):
        logger.debug(f"Model operator for {self._model_name} started")
        self._backend = model_backend
        while True:
            try:
                # collect input data
                kwargs = dict()
                for input in self._in:
                    data = input.get()
                    if not data:
                        break
                    for key, value in data.items():
                        if self._tokenizer and self._tokenizer.input == key:
                            msg_str = value.as_numpy()[0][0].decode()
                            kwargs[self._tokenizer.output] = self._tokenizer(msg_str)
                        else:
                            kwargs[key] = value
                    logger.debug(f"Input collected: {kwargs}")
                    for result in self._backend(**kwargs):
                        self._out[0].put(result)
            except Exception as e:
                logger.exception(e)

class Inference:
    """The base model that drives the inference flow"""
    def initialize(self, check_point_dir):
        self._operators = []
        self._inputs = []
        self._outputs = []
        input_names = [i.name for i in config.input]
        output_names = [i.name for i in config.output]
        # set up the inference flow
        for model_config in config.models:
            self._operators.append(ModelOperator(model_config, check_point_dir))
        # default flow for single model use case
        if  len(self._operators) == 1:
            operator = self._operators[0]
            self._inputs.append(operator.inject(input_names))
            self._outputs = operator.outputs
        self._executor = ThreadPoolExecutor(max_workers=len(self._operators))
        self._future = None

    def finalize(self):
        self._executor.shutdown()

    def _submit(self, op: ModelOperator, backend: ModelBackend):
        self._future = self._executor.submit(lambda: op.run(backend))