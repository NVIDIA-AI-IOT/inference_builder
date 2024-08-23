
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import List
from abc import ABC, abstractmethod
from typing import Callable
from common.config import config
from common.utils import get_logger
from transformers import AutoTokenizer

logger = get_logger(__name__)

class DataFlow:
    """A single input flow to a model"""
    def __init__(self, names: List):
        self._names = names
        self._queue = Queue()

    @property
    def names(self):
        return self._names

    def put(self, item):
        self._queue.put(item)

    def get(self):
        return self._queue.get()

class ModelBackend(ABC):
    """Interface for standardizing the model backend """
    @abstractmethod
    def __call__(self, **kwargs):
        pass

class ModelOperator:
    """An model operator runs a single model"""
    def __init__(self, model_config, check_point_dir):
        self._model_name = model_config.name
        self._check_point_dir = check_point_dir
        self._in = []
        self._out = [DataFlow([o.name for o in model_config.output])]
        self._running = False
        self._tokenizer = None
        if not model_config.is_missing(model_config, "tokenizer"):
            if model_config.tokenizer.type == "auto":
                self._tokenizer = AutoTokenizer.from_pretrained(
                    f"{self._check_point_dir}/{self._model_name}", use_fast=True, use_legacy=False
                )
                self._tokenizer.input_name = model_config.tokenizer.input_name
                self._tokenizer.output_name = model_config.tokernizer.output_name

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
        while True:
            # collect input data
            kwargs = dict()
            for input in self._in:
                data = input.get()
                if not data:
                    break
                for d, n in zip(data, input.names):
                    if self._tokenizer and self._tokenizer.input_name == n:
                        kwargs[self._tokenizer.output_name] = self._tokenizer(d)
                    kwargs[n] = d
                logger.debug(f"Input collected: {kwargs}")
            results = model_backend(**kwargs)
            self._out.put(results)


class Inference:
    """The base model that drives the inference flow"""
    def __init__(self, check_point_dir):
        self._operators = {}
        self._inputs = []
        input_names = [i.name for i in config.input]
        output_names = [i.name for i in config.output]
        # set up the inference flow
        for model_config in config.inference.models:
            self._operators[model_config.name] = ModelOperator(model_config, check_point_dir)
        for flow in config.inference.flow:
            source = flow[0]
            sink = flow[1]
            # input handling
            if (isinstance(source, list) and all(e in input_names for e in source)):
                operator = next((o for o in self._operators if o.name == sink), None)
                if operator is None:
                    raise Exception(f"Model not recognized: {sink}")
                self._inputs.append(operator.inject(source))
            # output handling
            if (isinstance(sink, list) and all(e in output_names for e in sink)):
                operator = next((o for o in self._operators if o.name == source), None)
                if operator is None:
                    raise Exception(f"Model not recognized: {source}")
                self._outputs =  operator.outputs
        # default flow for single model use case
        if not config.flow and len(self._operators) == 1:
            operator = self._operators[0]
            self._inputs = operator.inject(input_names)
            self._outputs = operator.outputs
        self._executor = ThreadPoolExecutor(max_workers=len(self._operators))
        self._future = None

    def submit(self, callable:Callable):
        self._future = self._executor.submit(callable)

    def finalize(self):
        self._executor.shutdown()