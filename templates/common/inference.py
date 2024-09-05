
import base64
from concurrent.futures import ThreadPoolExecutor
import os
from queue import Queue, Empty
import tempfile
from abc import ABC, abstractmethod
from typing import Callable
import uuid
from common.config import global_config
from common.utils import get_logger
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import numpy as np
from collections import namedtuple
import json
from pyservicemaker.utils import MediaChunk, MediaExtractor
import torch

logger = get_logger(__name__)

np_datatype_mapping = {
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

@dataclass
class Stop:
    reason: str

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

    def _process_base64_image(self, images: np.ndarray):
        image_list = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for image in images:
                image_id = str(uuid.uuid4())
                asset_path = os.path.join(temp_dir, image_id)
                # Shouldn't we decode the bytes directly?
                with open(asset_path, "wb") as f:
                    f.write(base64.b64decode(image.decode()))
                frame = MediaExtractor(chunks=[MediaChunk(asset_path)])()[0].get()
                image_list.append(torch.utils.dlpack.from_dlpack(frame.tensor))
        return image_list

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

    def put(self, item: Union[Dict, Error, Stop]):
        if not item:
            self._queue.put(item, timeout=self._timeout)
            return
        # collect data and put to the queue
        collected = dict()
        for name in item:
            tensor = item[name]
            idx, cfg = self.get_config(name)
            o_name = self._outputs[idx]
            if not cfg:
                logger.error(f"{name} is not a valid input of the flow")
                continue
            # handling custom data type
            data_type = cfg["data_type"]
            if  data_type not in np_datatype_mapping and isinstance(tensor, np.ndarray):
                logger.debug(f"Processing custom type: {data_type}")
                if data_type == "TYPE_CUSTOM_IMAGE_BASE64":
                    tensor = self._process_base64_image(tensor)
            collected[o_name] = tensor

        self._queue.put(collected, timeout=self._timeout)

    def get(self):
        try:
            item = self._queue.get(timeout=self._timeout)
        except Empty:
            item = Error("timeout")
        return item


class ModelBackend(ABC):
    """Interface for standardizing the model backend """
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise Exception("Not Implemented")

class Tokenizer:
    def __init__(self, config: Dict, check_point_dir, model_path: str):
        self._type = config['type'] if "type" in config else 'auto'
        self._model_path = config['model_path'] if 'model_path' in config else model_path
        self._encoder_config = config["encoder"]
        self._decoder_config = config["decoder"]
        self._tokenizer = None
        if self._type == 'auto':
            logger.info(f"Loading pretrained tokenizer from {model_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(check_point_dir, self._model_path), use_fast=True, use_legacy=False)

    @property
    def decoder_config(self):
        return self._decoder_config

    @property
    def encoder_config(self):
        return self._encoder_config

    def encode(self, *args):
        return self._tokenizer(*args)

    def decode(self, *args, **kwargs):
        return self._tokenizer.decode(args, kwargs)

class Preprocessor(ABC):
    def __init__(self, config: Dict):
        self._kind = config['kind'] if 'kind' in config else 'auto'
        self._input = config['input'] if 'input' in config else []
        self._output = config['output'] if 'output' in config else []
        self._preprocessor = None

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def kind(self):
        return self._kind

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

#############################################################################
# Below implementation should be provide while the package is being built
#############################################################################
class VilaMuxer:
    def __init__(self, model_path):
        llm_config_path = os.path.join(model_path, "fp16", "1-gpu", "config.json")
        with open(llm_config_path, "r") as f:
            config = json.load(f)
            self.vocab_size = config["pretrained_config"]["vocab_size"]

    def __call__(self, input_ids, features):
        image_embed_input_ids = np.arange(
                self.vocab_size,
                self.vocab_size + features.shape[0])
        extended_ids = np.append(input_ids, image_embed_input_ids)
        id_length = extended_ids.shape[0]
        vocab_size = features.shape[0]
        return extended_ids, np.array([id_length]), np.array([vocab_size]), features.cpu().numpy()

class VilaVisionEncoderProcessor:
    def __init__(self, model_path):
        from llava.model.multimodal_encoder.siglip.image_processing_siglip import SiglipImageProcessor
        self._preprocessor = SiglipImageProcessor.from_pretrained(model_path)

    def __call__(self, *args, **kwargs):
        # TODO value parser should be configurable
        return self._preprocessor(*args)['pixel_values'][0]
#####################################################################################################

class AutoPreprocessor(Preprocessor):
    """AutoPreprocessor loads the preprocessor from pretrained"""
    def __init__(self, config: Dict, check_point_dir: str, model_path: str):
        super().__init__(config)
        if "model_path" in config:
            model_path = config["model_path"]
        # TODO these should come from the registry or provided by user
        if self._kind == 'vila-visionenc':
            logger.info(f"Loading pretrained preprocessor from {model_path}")
            self._preprocessor = VilaVisionEncoderProcessor(os.path.join(check_point_dir, model_path))
        elif self._kind == 'vila-muxer':
            self._preprocessor = VilaMuxer(os.path.join(check_point_dir, model_path))

    def __call__(self, *args, **kwargs):
        # TODO value parser should be configurable
        return self._preprocessor(*args, **kwargs)


class ModelOperator:
    """An model operator runs a single model"""
    def __init__(self, model_config:Dict, check_point_dir: str):
        self._model_name = model_config["name"]
        self._check_point_dir = check_point_dir
        self._in: List[DataFlow] = []
        self._out: List[DataFlow] = []
        self._running = False
        self._tokenizer = None
        self._preprocessors = []
        if "tokenizer" in model_config:
            self._tokenizer = Tokenizer(
                config=model_config["tokenizer"],
                check_point_dir=self._check_point_dir,
                model_path=self._model_name
            )
        if "preprocessors" in model_config:
            for config in model_config["preprocessors"]:
                self._preprocessors.append(
                    AutoPreprocessor(
                        config=config,
                        check_point_dir=check_point_dir,
                        model_path=self._model_name
                    )
                )
        self._model_config = model_config
        self._backend = None

    @property
    def model_name(self):
        return self._model_name

    @property
    def outputs(self):
        return self._out.copy()

    def bind_input(self, configs: List[Dict], targets: List[str]=[]):
        if not targets:
            targets = [i['name'] for i in configs]
        flow = DataFlow(configs=configs, outputs=targets)
        self._in.append(flow)
        logger.debug(f"Data flow < {flow.in_names} -> {flow.o_names} > connected to model {self._model_name}")
        return flow

    def import_input(self, input: DataFlow):
        self._in.append(input)

    def export_output(self, outputs: List[str]=[], targets: List[str]=[]):
        if not outputs:
            outputs = [i["name"] for i in self._model_config["output"]]
        if not targets:
            targets = outputs
        configs = []
        for output in outputs:
            cfg = next((o for o in self._model_config["output"] if o["name"] == output), None)
            if cfg:
                configs.append(cfg)
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
                args = []
                kwargs = dict()
                for input in self._in:
                    data = input.get()
                    if not data:
                        # Not data, skip it
                        continue
                    kwargs.update(data)
                logger.debug(f"Input collected: {kwargs}")
                # apply tokenizer if required
                if self._tokenizer:
                    selected = self._tokenizer.encoder_config[0]
                    value = kwargs.pop(selected)
                    msg_str = [ v.decode() for v in value ]
                    tokenized = self._tokenizer.encode(*msg_str)
                    o_key = self._tokenizer.encoder_config[1]
                    kwargs[o_key] = np.array(tokenized[o_key])
                if all([isinstance(v, list) for _, v in kwargs.items()]):
                    # construct multiple inference requests
                    vs = [kwargs[k] for k in kwargs]
                    lengths = {len(i) for i in vs}
                    if len(lengths) == 1:
                        # construct multiple inference requests
                        l = lengths.pop()
                        args = [{k: kwargs[k][i] for k in kwargs } for i in range(l)]
                else:
                    args = [kwargs]
                # call preprocess() before passing args to the backend
                args = self._preprocess(args)
                # execute inference backend and collect result
                for result in self._backend(*args):
                    if isinstance(result, Error):
                        logger.error(f"Error in inferece: {result}")
                        break
                    # collect the result
                    for out in self._out:
                        if isinstance(result, Stop):
                            expected_result = result
                        else:
                            expected_result = dict()
                            for n in out.in_names:
                                if n in result:
                                    expected_result[n] = result[n]
                            if len(expected_result) != len(out.in_names):
                                logger.error(f"Not all the expected result is received from model {self._model_name}")
                                continue
                            if self._tokenizer:
                                expected_key = self._tokenizer.decoder_config[0]
                                value = expected_result.pop(expected_key, None)
                                if value is not None and isinstance(value, np.ndarray):
                                    # CPU only tokenizer for now
                                    value = value.flatten().tolist()
                                    text = self._tokenizer._tokenizer.decode(value, skip_special_tokens=True)
                                    expected_result[expected_key] = np.array([text], np.string_)
                                else:
                                    logger.error("Format not supported by tokenizer")
                        out.put(expected_result)
            except Exception as e:
                logger.exception(e)

    def _preprocess(self, args: List):
        logger.debug(f"Preprocessing data for model {self._model_name}")

        # go through the preprocess chain
        outcome = args
        for preprocessor in self._preprocessors:
            result = []
            for data in outcome:
                # initialize the processed as the original values
                processed = {k: v for k, v in data.items()}
                if not all([i in data for i in preprocessor.input]):
                    logger.warning(f"Input settings invalid for the preprocessor: {preprocessor.kind}")
                    continue
                input = [processed.pop(i) for i in preprocessor.input]
                logger.debug(f"{self._model_name} invokes preprocessor {preprocessor.kind} with given input {input}")
                output = preprocessor(*input)
                logger.debug(f"{self._model_name} preprocessor {preprocessor.kind} generated output {output}")
                if not isinstance(output, tuple):
                    output = (output,)
                if len(output) != len(preprocessor.output):
                    logger.warning(f"Number of preprocessing output doesn't match the configuration, expecting {len(preprocessor.output)}, while getting {len(output)}")
                    continue
                # update as processed
                for key, value in zip(preprocessor.output, output):
                    # data validation first
                    i_config = next((i for i in self._model_config['input'] if i["name"] == key), None)
                    if i_config is None:
                        logger.warning(f"Unexpected data: {key}")
                        continue
                    else:
                        data_type = i_config["data_type"]
                        if isinstance(value, np.ndarray):
                            data_type = np_datatype_mapping[data_type]
                            if value.dtype != data_type:
                                value = value.astype(data_type)
                    processed[key] = value
                result.append(processed)
            # update outcome
            outcome = result
        return outcome

class Inference:
    """The base model that drives the inference flow"""
    def initialize(self, check_point_dir: str):
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
                if len(s) == 2 and s[1]:
                    data = json.loads(s[1])
                    if isinstance(data, list):
                        d1 = data
                    else:
                        logger.error("Invalid routes")
                if len(t) == 2 and t[1]:
                    data = json.loads(t[1])
                    if isinstance(data, list):
                        d2 = data
                    else:
                        logger.error("Invalid routes")

            return Route(Path(m1, m2), Path(d1, d2))


        self._operators: ModelOperator = []
        self._inputs: List[DataFlow] = []
        self._outputs: List[DataFlow] = []
        self._input_config = OmegaConf.to_container(global_config.input)
        self._output_config = OmegaConf.to_container(global_config.output)

        # set up the inference flow
        for model_config in global_config.models:
            self._operators.append(ModelOperator(OmegaConf.to_container(model_config), check_point_dir))

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
                        # this is the model provides data
                        operator2 = next((o for o in self._operators if o.model_name == route.model.source), None)
                        if operator2 is None:
                            logger.error(f"Model {route.model.source} in the routes not found")
                            continue
                        o_flow = operator2.export_output(route.data.source, route.data.target)
                        operator.import_input(o_flow)
                    else:
                        # this is the top level input
                        if route.data.source:
                            configs = [
                                OmegaConf.to_container(c) for c in global_config.input if c.name in route.data.source
                            ]
                        else:
                            configs = [OmegaConf.to_container(c) for c in global_config.input]
                        self._inputs.append(operator.bind_input(configs, route.data.target))
                        pass
                elif route.model.source:
                    # this is the top level output
                    operator = next((o for o in self._operators if o.model_name == route.model.source), None)
                    if operator is None:
                        logger.error(f"Model {route.model.source} in the routes not found")
                        continue
                    self._outputs.append(operator.export_output(route.data.source, route.data.target))
                else:
                    logger.warning("Empty route entry")
        elif  len(self._operators) == 1:
            # default flow for single model use case without an explicit route
            operator = self._operators[0]
            # assume the input names all match the model
            self._inputs.append(operator.bind_input(OmegaConf.to_container(global_config.input)))
            self._outputs.append(operator.export_output())
        else:
            logger.error("Unable to set up inference routes")
        self._executor = ThreadPoolExecutor(max_workers=len(self._operators))
        self._future = None

    def finalize(self):
        self._executor.shutdown()

    def _submit(self, op: ModelOperator, backend: ModelBackend):
        self._future = self._executor.submit(lambda: op.run(backend))