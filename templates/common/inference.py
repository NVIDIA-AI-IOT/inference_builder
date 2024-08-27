
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from typing import List
from abc import ABC, abstractmethod
from typing import Callable
from common.config import global_config
from common.utils import get_logger
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from collections import namedtuple
import json

logger = get_logger(__name__)

datatype_mapping = {
    "TYPE_INVALID": None,
    "TYPE_BOOL": np.bool_,
    "TYPE_UINT8": np.ubyte,
    "TYPE_UINT16": np.uint16,
    "TYPE_UINT32": np.uint32,
    "TYPE_UINT64": np.uint64,
    "TYPE_INT8": np.byte,
    "TYPE_INT16": np.int16,
    "TYPE_INT32": np.int32,
    "TYPE_INT64": np.int64,
    "TYPE_FP16": np.float16,
    "TYPE_FP32": np.float32,
    "TYPE_FP64": np.float64,
    "TYPE_STRING": np.string_,
    "TYPE_BF16": None
}

@dataclass
class Error:
    message: str

    def __bool__(self):
        return False

Path = namedtuple('Path', ['source', 'target'])
Route = namedtuple('Route', ['model', 'data'])

class DataFlow:
    """A single input flow to a model"""
    def __init__(self, configs: List[Dict], outputs: List[str], timeout=None):
        self._configs = configs
        self._outputs = outputs
        self._queue = Queue()
        self._timeout = timeout

    def _transform(self, tensor, config: Dict):
        data_type = config["data_type"]
        np_datatype = datatype_mapping[data_type]
        if not np_datatype:
            logger.error(f"Invalid data type: {data_type}")
            return None

    @property
    def in_names(self):
        return [ i['name'] for i in self._configs]

    @property
    def o_names(self):
        return self._outputs

    def get_config(self, name:str) -> Tuple[int, Dict]:
        for i, config in enumerate(self._configs):
            if config["name"] == name:
                return i, config
        return -1, None

    def put(self, item: Dict):
        std_item = dict()
        for name, tensor in item.items():
            idx, cfg = self.get_config(name)
            o_name = self._outputs[idx]
            if not cfg:
                logger.error(f"{name} is not a valid input of the flow")
                continue
            if isinstance(tensor, np.ndarray):
                std_item[o_name] = tensor
            else:
                # need format conversion
                if hasattr(tensor, "as_numpy") and callable(tensor.as_numpy):
                    # implicit format conversion
                    transformed = tensor.as_numpy()
                    logger.debug(f"input {name} prepared as {type(transformed)}")
                else:
                    # explicit format conversion. TODO
                    transformed = self._transform(tensor, cfg)
                    logger.debug(f"input {name} tranformed to {o_name}: {type(tensor)} -> {type(transformed)}")
                std_item[o_name] = transformed
        self._queue.put(std_item, timeout=self._timeout)

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

class Tokenizer:
    def __init__(self, config: Dict, model_path: str):
        self._type = config['type'] if "type" in config else 'auto'
        self._encoder_config = config["encoder"]
        self._decoder_config = config["decoder"]
        self._tokenizer = None
        if self._type == 'auto':
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, use_legacy=False)

    @property
    def decoder_config(self):
        return self._decoder_config

    @property
    def encoder_config(self):
        return self._encoder_config

    def encode(self, *args):
        return self._tokenizer(args)

    def decode(self, *args, **kwargs):
        return self._tokenizer.decode(args, kwargs)

class ModelOperator:
    """An model operator runs a single model"""
    def __init__(self, model_config, check_point_dir):
        self._model_name = model_config.name
        self._check_point_dir = check_point_dir
        self._in = []
        self._out = []
        self._running = False
        self._tokenizer = None
        if hasattr(model_config, "tokenizer"):
            self._tokenizer = Tokenizer(
                config=OmegaConf.to_container(model_config.tokenizer),
                model_path=f"{self._check_point_dir}/{self._model_name}"
            )
        self._model_config = model_config

    @property
    def model_name(self):
        return self._model_name

    @property
    def outputs(self):
        return self._out.copy()

    def inject(self, configs: List[Dict], targets: List[str]=[]):
        if not targets:
            targets = [i['name'] for i in configs]
        flow = DataFlow(configs=configs, outputs=targets)
        self._in.append(flow)
        logger.debug(f"Data flow < {flow.in_names} -> {flow.o_names} > connected to model {self._model_name}")
        return flow

    def export(self, outputs: List[str]=[], targets: List[str]=[]):
        if not outputs:
            outputs = [i.name for i in self._model_config.output]
        if not targets:
            targets = outputs
        configs = []
        for output in outputs:
            cfg = next((o for o in self._model_config.output if o.name == output), None)
            if cfg:
                configs.append(OmegaConf.to_container(cfg))
        flow = DataFlow(configs, targets)
        logger.debug(f"Data flow < {flow.in_names} -> {flow.o_names} > connected to model {self._model_name}")
        self._out.append(flow)
        return flow

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
                        logger.error(f"Error getting input data: {data}")
                        continue
                    logger.debug(f"Input collected: {data}")
                    for key, value in data.items():
                        if self._tokenizer and self._tokenizer.encoder_config[0] == key:
                            msg_str = value[0][0].decode()
                            tokenized = self._tokenizer.encode(msg_str)
                            o_key = self._tokenizer.encoder_config[1][0]
                            len_key = self._tokenizer.encoder_config[1][1]
                            kwargs[o_key] = np.array(tokenized[o_key], dtype=np.int32)
                            kwargs[len_key] = np.array([[len(tokenized)]], dtype=np.int32)
                        else:
                            kwargs[key] = value
                    for result in self._backend(**kwargs):
                        if not result:
                            logger.error(f"Error in inferece: {result}")
                            break
                        # collect the result
                        for out in self._out:
                            expected_result = dict()
                            for n in out.in_names:
                                if n in result:
                                    expected_result[n] = result[n]
                            if len(expected_result) != len(out.in_names):
                                logger.error(f"Not all the expected result is received")
                                continue
                            if not self._tokenizer:
                                out.put(expected_result)
                            else:
                                expected_key = self._tokenizer.decoder_config[0]
                                value = result.pop(expected_key, None)
                                if value is not None and hasattr(value, 'as_numpy') and callable(value.as_numpy):
                                    value = value.as_numpy().flatten().tolist()
                                    text = self._tokenizer._tokenizer.decode(value, skip_special_tokens=True)
                                    expected_result[expected_key] = np.array([text], np.object_)
                                out.put(expected_result)

            except Exception as e:
                logger.exception(e)

class Inference:
    """The base model that drives the inference flow"""
    def initialize(self, check_point_dir):
        def parse_route(i1, i2) -> Route:
            m1 = ''
            m2 = ''
            d1 = []
            d2 = []
            s = i1.split(':')
            t = i2.split(':')
            if len(s) > 2 or len(t) > 2:
                logger.error("Invalid routes")
            else:
                m1 = s[0]
                m2 = t[0]
                if len(s) == 2 and s[1] != '*':
                    data = json.loads(s[1])
                    if isinstance(data, list):
                        d1 = data
                    else:
                        logger.error("Invalid routes")
                if len(t) == 2 and t[1] != '*':
                    data = json.loads(t[1])
                    if isinstance(data, list):
                        d2 = data
                    else:
                        logger.error("Invalid routes")

            return Route(Path(m1, m2), Path(d1, d2))


        self._operators: ModelOperator = []
        self._inputs: List[DataFlow] = []
        self._outputs: List[DataFlow] = []
        input_names = [i.name for i in global_config.input]
        output_names = [i.name for i in global_config.output]
        # set up the inference flow
        for model_config in global_config.models:
            self._operators.append(ModelOperator(model_config, check_point_dir))
            if hasattr(global_config, "routes"):
                # go through the routing table
                for k, v in global_config.routes.items():
                    route = parse_route(k, v)
                    logger.debug(f"Adding route {route}")
                    if route.model.target:
                        operator = next((o for o in self._operators if o.model_name == route.model.target), None)
                        if operator is None:
                            logger.error(f"Model {route.model.target} in the routes not found")
                            continue
                        if route.model.source:
                            # TODO
                            pass
                        else:
                            # this is the top level input
                            if route.data.source:
                                configs = [
                                    OmegaConf.to_container(c) for c in global_config.input if c.name in route.data.source
                                ]
                            else:
                                configs = [OmegaConf.to_container(c) for c in global_config.input]
                            self._inputs.append(operator.inject(configs, route.data.target))
                            pass
                    elif route.model.source:
                        # this is the top level output
                        operator = next((o for o in self._operators if o.model_name == route.model.source), None)
                        if operator is None:
                            logger.error(f"Model {route.model.source} in the routes not found")
                            continue
                        self._outputs.append(operator.export(route.data.source, route.data.target))
                    else:
                        logger.warning("Empty route entry")
            elif  len(self._operators) == 1:
                # default flow for single model use case without an explicit route
                operator = self._operators[0]
                # assume the input names all match the model
                self._inputs.append(operator.inject(OmegaConf.to_container(global_config.input)))
                self._outputs.append(operator.export())
            else:
                logger.error("Unable to set up inference routes")
        self._executor = ThreadPoolExecutor(max_workers=len(self._operators))
        self._future = None

    def finalize(self):
        self._executor.shutdown()

    def _submit(self, op: ModelOperator, backend: ModelBackend):
        self._future = self._executor.submit(lambda: op.run(backend))