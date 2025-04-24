
import base64
from concurrent.futures import ThreadPoolExecutor
import os
from queue import Queue, Empty
from abc import ABC, abstractmethod
from config import global_config
from .utils import get_logger, split_tensor_in_dict
from .codec import ImageDecoder
import custom
import transformers
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import numpy as np
from collections import namedtuple
import json
from pyservicemaker import Pipeline, as_tensor
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
    "TYPE_CUSTOM_DS_IMAGE": np.ubyte,
    "TYPE_CUSTOM_DS_MIME": np.string_,
    "TYPE_CUSTOM_DS_PASSTHROUGH": None,
    "TYPE_BF16": None
}

torch_datatype_mapping = {
    "TYPE_INVALID": None,
    "TYPE_BOOL": torch.bool,
    "TYPE_UINT8": torch.int8,
    "TYPE_UINT16": torch.int16,
    "TYPE_UINT32": torch.int32,
    "TYPE_UINT64": torch.int64,
    "TYPE_INT8": torch.int8,
    "TYPE_INT16": torch.int16,
    "TYPE_INT32": torch.int32,
    "TYPE_INT64": torch.int64,
    "TYPE_FP16": torch.float16,
    "TYPE_FP32": torch.float32,
    "TYPE_FP64": torch.float64,
    "TYPE_STRING": None,
    "TYPE_CUSTOM_DS_IMAGE": torch.int8,
    "TYPE_CUSTOM_DS_MIME": None,
    "TYPE_CUSTOM_DS_PASSTHROUGH": None,
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
    """A single data flow from or to a model"""
    def __init__(
            self,
            configs: List[Dict],
            tensor_names: List[Tuple[str, str]],
            inbound: bool = False,
            outbound: bool = False,
            timeout=None
        ):
        self._configs = configs
        self._tensor_names = tensor_names
        self._inbound = inbound
        self._outbound = outbound
        self._timeout = timeout
        self._queue = Queue()

    def _process_custom_data(self, tensor: np.ndarray, data_type: str):
        processed = tensor
        if self._inbound:
            if data_type == "TYPE_CUSTOM_BINARY_URLS":
                processed = [input for input in tensor]
            elif data_type == "TYPE_CUSTOM_BINARY_BASE64":
                processed = []
                for input in tensor:
                    if isinstance(input, np.bytes_) or isinstance(input, bytes):
                        input = input.decode()
                    elif not isinstance(input, np.str_):
                        logger.error(f"base64 binary must be bytes or string")
                        continue
                    processed.append(np.frombuffer(base64.b64decode(input), dtype=np.uint8))
        return processed

    @property
    def in_names(self):
        return [i[0] for i in self._tensor_names]

    @property
    def o_names(self):
        return [i[1] for i in self._tensor_names]

    def get_config(self, name: str):
        if not self._configs:
            return None
        return next((config for config in self._configs if config["name"] == name), None)

    def put(self, item: Union[Dict, Error, Stop]):
        if not item:
            # empty item is used to signal the end of the data flow
            logger.info("Data flow ended with empty item")
            self._queue.put(item, timeout=self._timeout)
            return
        # check input data integrity
        if self._inbound:
            for config in self._configs:
                name = config["name"]
                optional = "optional" in config and config["optional"]
                if name not in item and not optional:
                    logger.error(f"{name} is not optional and not found from the dataflow input configs!")
                    return
        # collect data and deposit it to the queue
        collected = {}
        for i_name, o_name in self._tensor_names:
            if i_name not in item:
                continue
            tensor = item[i_name]
            # handling custom data type
            if self._inbound or self._outbound:
                config = self.get_config(i_name)
                if config and not config["data_type"] in np_datatype_mapping and isinstance(tensor, np.ndarray):
                    logger.debug(f"Processing custom data type: {config['data_type']}")
                    collected[o_name] = self._process_custom_data(tensor, config["data_type"])
                else:
                    collected[o_name] = tensor
            else:
                collected[o_name] = tensor
        # check output data integrity
        if self._outbound:
            for config in self._configs:
                name = config["name"]
                optional = "optional" in config and config["optional"]
                if name not in collected and not optional:
                    logger.error(f"{name} is not optional and not found from the dataflow output configs!")
                    return
        if collected:
            self._queue.put(collected, timeout=self._timeout)
        else:
            logger.error(f"No data to deposit to the queue")

    def get(self):
        try:
            item = self._queue.get(timeout=self._timeout)
        except Empty:
            item = Error("timeout")
        return item

class ImageInputDataFlow(DataFlow):
    """A data flow for image data"""
    def __init__(self, configs: List[Dict], tensor_names: List[Tuple[str, str]], timeout=None):
        super().__init__(configs, tensor_names, True, False,timeout)
        self._image_decoder = ImageDecoder(["JPEG", "PNG"])

    def _process_custom_data(self, images: np.ndarray, data_type: str):
        if self._image_decoder is None:
            return Error("Image decoder is not sucessfully created")
        if data_type == "TYPE_CUSTOM_BASE64_IMAGE":
            return self._process_base64_image(images)
        else:
            return super()._process_custom_data(images, data_type)

    def _process_base64_image(self, images: np.ndarray):
        result = []
        for image in images:
            if isinstance(image, np.bytes_) or isinstance(image, bytes):
                image = image.decode()
            elif not isinstance(image, np.str_):
                logger.error(f"base64 image must be bytes or string: {type(image)}")
                continue
            data_prefix, data_payload = image.split(",")
            mime_type = data_prefix.split(";")[0].split(":")[1]
            format = None
            if mime_type == "image/jpeg" or mime_type == "image/jpg":
                format = "JPEG"
            elif mime_type == "image/png":
                format = "PNG"
            else:
                raise ValueError(f"Unsupported image format: {mime_type}")
            data_payload = base64.b64decode(data_payload)
            tensor = as_tensor(np.frombuffer(data_payload, dtype=np.uint8).copy(), format)
            result.append(self._image_decoder.decode(tensor, format))
        return result

inbound_dataflow_mapping = {
    "TYPE_CUSTOM_IMAGE_BASE64": ImageInputDataFlow,
}
class ModelBackend(ABC):
    """Interface for standardizing the model backend """
    def __init__(self, model_config: Dict, device_id=0):
        self._model_config = model_config
        self._device_id = device_id

    @property
    def device_id(self):
        return self._device_id

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise Exception("Not Implemented")

    def stop(self):
        logger.info(f'Backend for {self._model_config["name"]} stopped')

class Processor(ABC):
    def __init__(self, config: Dict):
        self._name = config['name']
        self._kind = config['kind'] if 'kind' in config else 'auto'
        self._input = config['input'] if 'input' in config else []
        self._output = config['output'] if 'output' in config else []
        self._config = { 'device_id': 0 }
        if 'config' in config:
            self._config.update(config['config'])
        self._name = config['name']
        self._processor = None

    @property
    def name(self):
        return self._name

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def kind(self):
        return self._kind

    @property
    def config(self):
        return self._config

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class AutoProcessor(Processor):
    """AutoPrrocessor loads the preprocessor from pretrained"""
    def __init__(self, config: Dict, check_point_dir: str, model_path: str):
        super().__init__(config)
        if "model_path" in config:
            model_path = config["model_path"]
        self._processor = transformers.AutoProcessor.from_pretrained(os.path.join(check_point_dir, model_path))
        if self._processor is None:
            logger.error(f"Failed to load AutoProcessor from {model_path}")

    def __call__(self, *args):
        # TODO value parser should be configurable
        return self._processor(*args)

class CustomProcessor(Processor):
    """CustomProcessor loads the processor from custom module"""
    def __init__(self, config: Dict, check_point_dir: str, model_path: str):
        super().__init__(config)
        if "model_path" not in self.config:
            self.config["model_path"] = model_path
        self.config["model_path"] = os.path.join(check_point_dir, self.config["model_path"])
        self._processor = custom.create_instance(self.name, self.config)
        if self._processor is not None:
            logger.info(f"Custom processor {self._processor.name} created")
        else:
            logger.error(f"Failed to create processor {self.name}")

    def __call__(self, *args):
        ret = self._processor(*args)
        if not isinstance(ret, tuple):
            ret = ret,
        return ret

class ModelOperator:
    """An model operator runs a single model"""
    def __init__(self, model_config:Dict, check_point_dir: str):
        self._model_name = model_config["name"]
        self._check_point_dir = check_point_dir
        self._in: List[DataFlow] = []
        self._out: List[DataFlow] = []
        self._running = False
        self._preprocessors = []
        self._postprocessors = []
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
        flow = None
        tensor_names = [(i['name'], o) for i, o in zip(configs, targets)]
        for config in configs:
            if config["data_type"] in inbound_dataflow_mapping:
                # customized inbound data flow
                flow = inbound_dataflow_mapping[config["data_type"]](configs, tensor_names)
                break
        if flow is None:
            tensor_names = [(i['name'], o) for i, o in zip(configs, targets)]
            flow = DataFlow(configs, tensor_names, inbound=True)
        self._in.append(flow)
        logger.debug(f"Data flow < {flow.in_names} -> {flow.o_names} > connected to model {self._model_name}")
        return flow

    def bind_output(self, configs: List[Dict], sources: List[str]=[]):
        if not sources:
            sources = [i['name'] for i in configs]
        tensor_names = [(i, o['name']) for i, o in zip(sources, configs)]
        flow = DataFlow(configs, tensor_names, outbound=True)
        self._out.append(flow)
        logger.debug(f"model {self._model_name} connected to Data flow < {flow.in_names} -> {flow.o_names} >")
        return flow

    def import_input(self, input: DataFlow):
        self._in.append(input)

    def import_output(self, output: DataFlow):
        self._out.append(output)

    def run(self, model_backend: ModelBackend):
        logger.debug(f"Model operator for {self._model_name} started")

        # create preprocessors
        for kind, processors in [("preprocessors", self._preprocessors), ("postprocessors", self._postprocessors)]:
            if kind in self._model_config:
                for config in self._model_config[kind]:
                    ProcessorClass = None
                    if config["kind"] == "auto":
                        ProcessorClass = AutoProcessor
                    elif config["kind"] == "custom":
                        ProcessorClass = CustomProcessor
                    if ProcessorClass is not None:
                        processors.append(
                            ProcessorClass(
                                config=config,
                                check_point_dir=self._check_point_dir,
                                model_path=self._model_name
                            )
                        )
                    else:
                        raise Exception("Invalid Processor")
        # backend loop
        self._backend = model_backend
        while True:
            try:
                # collect input data until Stop is received
                args = []
                kwargs = dict()
                for input in self._in:
                    data = input.get()
                    if not data:
                        # Not data, skip it
                        continue
                    kwargs.update(data)
                if not kwargs:
                    continue
                message = "Completed"
                logger.debug(f"Input collected: {kwargs}")
                if any([isinstance(v, list) for _, v in kwargs.items()]):
                    # construct multiple inference requests
                    args = split_tensor_in_dict(kwargs)
                    kwargs = {}
                # call preprocess() before passing args to the backend
                processed = self._preprocess(args if args else [kwargs])
                if args:
                    # sequential processed data
                    args = processed
                elif kwargs:
                    # batch processed data
                    kwargs = processed[0]
                else:
                    logger.error(f"Invalid result from preprocess: {args} and {kwargs}")
                    continue
                # execute inference backend and collect result
                for r in self._backend(*args, **kwargs):
                    if not self._out:
                        logger.error(f"No output data flow is bound to model {self._model_name}, please check the route configuration")
                        continue
                    # iterate the result list and postprocess each of them
                    for out in self._out:
                        output_data = {n : [] for n in out.in_names}
                        if isinstance(r, list):
                            # we get a batch
                            for result in r:
                                result = self._postprocess(result)
                                if not all([n in result for n in out.in_names]):
                                    logger.error(f"Not all the expected result is received from model {self._model_name}:{out.in_names}")
                                    continue
                                # collect the result
                                for n, v in output_data.items():
                                    if n in result:
                                        v.append(result[n])
                                    else:
                                        v.append(None)
                        else:
                            output_data = self._postprocess(r)
                            if not all([n in output_data for n in out.in_names]):
                                logger.error(f"Not all the expected result is received from model {self._model_name}:{out.in_names}")
                                continue
                        logger.debug(f"Deposit result: {output_data}")
                        out.put(output_data)

            except Exception as e:
                logger.exception(e)
                message = "Error"
            # notify the end of the current inference cycle
            for out in self._out:
                out.put(Stop(message))

    def _preprocess(self, args: List):
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
                    logger.error("Return value of a processor must be a tuple")
                    continue
                if len(output) != len(preprocessor.output):
                    logger.warning(f"Number of preprocessing output doesn't match the configuration, expecting {len(preprocessor.output)}, while getting {len(output)}")
                    continue
                # update as processed
                for key, value in zip(preprocessor.output, output):
                    processed[key] = value
                result.append(processed)
            # update outcome
            outcome = result
        # correct the data type
        for data in outcome:
            for key in data:
                value = data[key]
                i_config = next((i for i in self._model_config['input'] if i["name"] == key), None)
                if i_config is None:
                    logger.warning(f"Unexpected data: {key}")
                    continue
                else:
                    data_type = i_config["data_type"]
                    if isinstance(value, np.ndarray):
                        data_type = np_datatype_mapping[data_type]
                        if value.dtype != data_type:
                            data[key] = value.astype(data_type)
                    elif isinstance(value, torch.Tensor):
                        data_type = torch_datatype_mapping[data_type]
                        if value.dtype != data_type:
                            data[key] = value.to(data_type)
        return outcome

    def _postprocess(self, data: Dict):
        processed = {k: v for k, v in data.items()}
        for processor in self._postprocessors:
            if not all([i in data for i in processor.input]):
                logger.warning(f"Input settings invalid for the processor: {processor}")
                continue
            input = [processed.pop(i) for i in processor.input]
            logger.debug(f"Post-processor {processor.name} invoked with given input {input}")
            output = processor(*input)
            logger.debug(f"Post-processor generated output {output}")
            if len(output) != len(processor.output):
                logger.warning(f"Number of postprocessing output doesn't match the configuration, expecting {len(processor.output)}, while getting {len(output)}")
                continue
            # update as processed
            for key, value in zip(processor.output, output):
                processed[key] = value
        return processed

class InferenceBase:
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
                # neither source nor target model is specified.
                if not route.model.source and not route.model.target:
                    # this is a direct passthrough in the top level, we can use a standalone dataflow
                    if not route.data.source and not route.data.target:
                        logger.error(f"Invalid route: {route}, source or target is required for a direct pass")
                        continue
                    dataflow = None
                    if route.data.source:
                        s_configs = [OmegaConf.to_container(c) for c in global_config.input if c.name in route.data.source]
                        if len(configs) != len(route.data.source):
                            logger.error(f"Not all the sources are found in the input configs, unable to create passthrough dataflow")
                            continue
                        tensor_names = route.data.target if route.data.target else route.data.source
                        for n in tensor_names:
                            if n not in [c.name for c in global_config.output]:
                                logger.error(f"Output {n} not found in the output configs, unable to create passthrough dataflow")
                                continue
                        tensor_names = [(i['name'], o) for i, o in zip(s_configs, tensor_names)]
                        dataflow = DataFlow(configs=s_configs, tensor_names=tensor_names, inbound=True)
                    elif route.data.target:
                        t_configs = [OmegaConf.to_container(c) for c in global_config.output if c.name in route.data.target]
                        if len(t_configs) != len(route.data.target):
                            logger.error(f"Not all the targets are found in the output configs, unable to create passthrough dataflow")
                            continue
                        for n in route.data.target:
                            if n not in [c.name for c in global_config.input]:
                                logger.error(f"Input {n} not found in the input configs, unable to create passthrough dataflow")
                                continue
                        tensor_names = [(i, i) for i in route.data.target]
                        dataflow = DataFlow(configs=t_configs, tensor_names=tensor_names, outbound=True)
                    else:
                        logger.error(f"Invalid route: {route}, source or target is required for a direct pass")
                        continue
                    if dataflow is not None:
                        self._inputs.append(dataflow)
                        self._outputs.append(dataflow)
                elif route.model.target:
                    operator1 = next((o for o in self._operators if o.model_name == route.model.target), None)
                    if operator1 is None:
                        logger.error(f"Model {route.model.target} in the routes not found")
                        continue
                    # both source and target model are specified.
                    if route.model.source:
                        # this is the model provides data
                        operator2 = next((o for o in self._operators if o.model_name == route.model.source), None)
                        if operator2 is None:
                            logger.error(f"Model {route.model.source} in the routes not found")
                            continue
                        flow = None
                        if route.data.source and route.data.target:
                            if route.data.source != route.data.target:
                                logger.error(f"Source and target are different, unable to bind output for model {operator1.model_name}, {route.data.source} -> {route.data.target}")
                                continue
                            tensor_names = [(i, o) for i, o in zip(route.data.source, route.data.target)]
                            flow = DataFlow(configs=None, tensor_names=tensor_names)
                        elif route.data.source:
                            tensor_names = [(i, i) for i in route.data.source]
                            flow = DataFlow(configs=None, tensor_names=tensor_names)
                        elif route.data.target:
                            tensor_names = [(i, i) for i in route.data.target]
                            flow = DataFlow(configs=None, tensor_names=tensor_names)
                        else:
                            logger.error(f"Invalid route: {route}, source or target is required for connecting two models, {operator2.model_name} and {operator1.model_name}")
                            continue
                        operator2.import_output(flow)
                        operator1.import_input(flow)
                    else:
                        # this is the top level input
                        if route.data.source:
                            configs = [
                                OmegaConf.to_container(c) for c in global_config.input if c.name in route.data.source
                            ]
                        else:
                            configs = [OmegaConf.to_container(c) for c in global_config.input]
                        if route.data.target and route.data.target != [c.name for c in configs]:
                            logger.error(f"Source and target are different, unable to bind input for model {operator1.model_name}, {route.data.target} vs {configs}")
                            continue
                        self._inputs.append(operator1.bind_input(configs, route.data.target))
                # only source model is specified.
                elif route.model.source:
                    # this is the top level output
                    operator = next((o for o in self._operators if o.model_name == route.model.source), None)
                    if operator is None:
                        logger.error(f"Model {route.model.source} in the routes not found")
                        continue
                    if route.data.target:
                        configs = [OmegaConf.to_container(c) for c in global_config.output if c.name in route.data.source]
                    else:
                        configs = [OmegaConf.to_container(c) for c in global_config.output]
                    if route.data.source and route.data.source != [c['name'] for c in configs]:
                        logger.warning(f"Source and target are different for the output of {operator.model_name}, remember to add top level postprocessor")
                        dataflow = DataFlow(configs=None, tensor_names=[(i, i) for i in route.data.source])
                        operator.import_output(dataflow)
                        self._outputs.append(dataflow)
                    else:
                        self._outputs.append(operator.bind_output(configs))
                else:
                    logger.warning("Empty route entry")
        elif  len(self._operators) == 1:
            # default flow for single model use case without an explicit route
            operator = self._operators[0]
            # direct connection from top level input to the model
            self._inputs.append(operator.bind_input(OmegaConf.to_container(global_config.input)))
            # direct connection from the model to top level output
            self._outputs.append(operator.bind_output(OmegaConf.to_container(global_config.output)))
        else:
            logger.error("Unable to set up inference routes")
        self._executor = ThreadPoolExecutor(max_workers=len(self._operators))
        self._future = None
        # sanity check on vision pipeline
        try:
            Pipeline("vision")
        except Exception as e:
            logger.exception(e)

    def finalize(self):
        self._executor.shutdown()

    def _submit(self, op: ModelOperator, backend: ModelBackend):
        self._future = self._executor.submit(lambda: op.run(backend))