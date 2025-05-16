import base64
from concurrent.futures import ThreadPoolExecutor
import os
import types
import threading
from queue import Queue, Empty
from abc import ABC, abstractmethod
from config import global_config
from .utils import get_logger, split_tensor_in_dict
from .codec import ImageDecoder
import custom
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import numpy as np
from collections import namedtuple
import json
from pyservicemaker import Pipeline, as_tensor
from pyservicemaker.utils import MediaExtractor, MediaChunk
import torch
from .asset_manager import AssetManager

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
    "TYPE_UINT8": torch.uint8,
    "TYPE_UINT16": torch.uint16,
    "TYPE_UINT32": torch.uint32,
    "TYPE_UINT64": torch.uint64,
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
                logger.debug(f"DataFlow _process_custom_data: {data_type}")
                processed = [input for input in tensor]
            elif data_type == "TYPE_CUSTOM_BINARY_BASE64":
                logger.debug(f"DataFlow _process_custom_data: {data_type}")
                processed = []
                for input in tensor:
                    if isinstance(input, np.bytes_) or isinstance(input, bytes):
                        input = input.decode()
                    elif not isinstance(input, np.str_):
                        logger.error(f"base64 binary must be bytes or string")
                        continue
                    processed.append(np.frombuffer(base64.b64decode(input), dtype=np.uint8))
        return processed

    def _is_collected_valid(self, collected: Dict):
        if not collected:
            logger.info(f"No data collected from the dataflow: {self.in_names}")
            return False
        # check output data integrity
        if self._outbound:
            for config in self._configs:
                name = config["name"]
                optional = "optional" in config and config["optional"]
                if name not in collected and not optional:
                    logger.error(f"{name} is not optional and not found from the dataflow output configs!")
                    return False
        return True

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
            # pass Error or Stop to the downstream
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
                    collected[o_name] = self._process_custom_data(tensor, config["data_type"])
                else:
                    collected[o_name] = tensor
            else:
                collected[o_name] = tensor
        # check output data integrity
        if not self._is_collected_valid(collected):
            logger.warning(f"Invalid data collected from the dataflow: {self.in_names}, data: {collected}, ignored")
            return

        #  deposit the collected data to the queue
        generators = {}
        values = {}
        for k, v in collected.items():
            if isinstance(v, types.GeneratorType):
                generators[k] = v
            else:
                values[k] = v
        if generators:
            keys = list(generators.keys())
            generators = list(generators.values())
            for vs in zip(*generators):
                result = dict(zip(keys, vs))
                result.update(values)
                self._queue.put(result, timeout=self._timeout)
        else:
            self._queue.put(values, timeout=self._timeout)

    def get(self):
        try:
            item = self._queue.get(timeout=self._timeout)
        except Empty:
            item = Error("timeout")
        return item

class VideoInputDataFlow(DataFlow):
    """A data flow for video data"""
    def __init__(self, configs: List[Dict], tensor_names: List[Tuple[str, str]], key_tensor_type: str, timeout=None):
        super().__init__(configs, tensor_names, True, False, timeout)
        self._media_extractor = None
        self._video_tensor_type = key_tensor_type

    def _process_custom_data(self, tensor: np.ndarray, data_type: str):
        logger.debug(f"VideoInputDataFlow._process_custom_data: {data_type}")
        if data_type == "TYPE_CUSTOM_VIDEO_ASSETS":
            return self._process_video_assets(tensor)
        else:
            return super()._process_custom_data(tensor, data_type)

    def _is_collected_valid(self, collected: Dict):
        result = super()._is_collected_valid(collected)
        if not result:
            return False
        return any([isinstance(v, types.GeneratorType) for k, v in collected.items()])

    def _process_video_assets(self, assets: np.ndarray):
        urls = []
        duration = -1
        for asset in assets:
            asset_manager = AssetManager()
            asset = asset_manager.get_asset(asset)
            if asset:
                urls.append(asset.path)
                if duration == -1:
                    duration = asset.duration
                elif duration != asset.duration:
                    logger.error(f"Batched video assets must have the same duration!")
                    return
        if not urls:
            logger.error(f"No video assets found: {assets}")
            return
        self._media_extractor = MediaExtractor([MediaChunk(url, duration=duration) for url in urls])
        qs = self._media_extractor()
        try:
            while True:
                data = []
                for q in qs:
                    frame = q.get(timeout=1.0)
                    if frame is None:
                        logger.info(f"Duration reached: {assets}")
                        break
                    data.append(frame.tensor)
                yield data
        except Empty:
            logger.info(f"EOS on videos: {assets}")
        self._media_extractor = None

class ImageInputDataFlow(DataFlow):
    """A data flow for image data"""
    def __init__(self, configs: List[Dict], tensor_names: List[Tuple[str, str]], key_tensor_type: str,timeout=None):
        super().__init__(configs, tensor_names, True, False, timeout)
        self._image_decoder = ImageDecoder(["JPEG", "PNG"])
        self._image_tensor_names = []
        for tensor_name in tensor_names:
            config = next((c for c in configs if c["name"] == tensor_name[0]), None)
            if config and config["data_type"] == key_tensor_type:
                self._image_tensor_names.append(tensor_name[1])
        if not self._image_tensor_names:
            raise Exception("Image tensor name not found in the ImageInputDataFlow")

    def _process_custom_data(self, images: np.ndarray, data_type: str):
        logger.debug(f"ImageInputDataFlow._process_custom_data: {data_type}")
        if self._image_decoder is None:
            return Error("Image decoder is not sucessfully created")
        if data_type == "TYPE_CUSTOM_IMAGE_BASE64":
            return self._process_base64_image(images)
        elif data_type == "TYPE_CUSTOM_IMAGE_ASSETS":
            return self._process_image_assets(images)
        else:
            return super()._process_custom_data(images, data_type)

    def _is_collected_valid(self, collected: Dict):
        result = super()._is_collected_valid(collected)
        if not result:
            return False
        return all([n in collected for n in self._image_tensor_names])

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
                logger.error(f"Unsupported image format: {mime_type}")
                continue
            data_payload = base64.b64decode(data_payload)
            tensor = as_tensor(np.frombuffer(data_payload, dtype=np.uint8).copy(), format)
            result.append(self._image_decoder.decode(tensor, format))
        logger.debug(f"ImageInputDataFlow._process_base64_image generates {len(result)} tensors")
        return result

    def _process_image_assets(self, assets: np.ndarray):
        result = []
        for asset in assets:
            asset_manager = AssetManager()
            asset = asset_manager.get_asset(asset)
            if not asset:
                logger.error(f"Asset not found: {asset}")
                continue
            format = None
            if asset.mime_type == "image/jpeg" or asset.mime_type == "image/jpg":
                format = "JPEG"
            elif asset.mime_type == "image/png":
                format = "PNG"
            else:
                logger.error(f"Unsupported image format: {asset.mime_type}")
                continue
            with open(asset.path, "rb") as f:
                data = f.read()
                tensor = as_tensor(np.frombuffer(data, dtype=np.uint8).copy(), format)
                result.append(self._image_decoder.decode(tensor, format))
        return result

inbound_dataflow_mapping = {
    "TYPE_CUSTOM_IMAGE_BASE64": ImageInputDataFlow,
    "TYPE_CUSTOM_IMAGE_ASSETS": ImageInputDataFlow,
    "TYPE_CUSTOM_VIDEO_ASSETS": VideoInputDataFlow,
}
class ModelBackend(ABC):
    """Interface for standardizing the model backend """
    def __init__(self, model_config: Dict, model_home: str, device_id=0):
        self._model_config = model_config
        self._model_home = model_home
        self._device_id = device_id

    @property
    def device_id(self):
        return self._device_id

    @property
    def model_home(self):
        return self._model_home

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise Exception("Not Implemented")

    def stop(self):
        logger.info(f'Backend for {self._model_config["name"]} stopped')

class Processor(ABC):
    def __init__(self, config: Dict, model_home: str):
        self._name = config['name']
        self._kind = config['kind'] if 'kind' in config else 'auto'
        self._input = config['input'] if 'input' in config else []
        self._output = config['output'] if 'output' in config else []
        self._config = { 'device_id': 0, 'model_home': model_home }
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
    def __init__(self, config: Dict, model_home: str):
        super().__init__(config, model_home)
        import transformers
        self._processor = transformers.AutoProcessor.from_pretrained(model_home)
        if self._processor is None:
            logger.error(f"Failed to load AutoProcessor from {model_home}")

    def __call__(self, *args):
        # TODO value parser should be configurable
        return self._processor(*args)

class CustomProcessor(Processor):
    """CustomProcessor loads the processor from custom module"""
    def __init__(self, config: Dict, model_home: str):
        super().__init__(config, model_home)
        if not hasattr(custom, "create_instance"):
            raise Exception("Custom processor module not valid!!")
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

class Collector(ABC):
    """Collector is an interface for collecting data from the data flow"""
    def collect(self):
        raise NotImplementedError("Not implemented")

class SingleFlowCollector(Collector):
    """SingleFlowCollector is a collector for a single data flow"""
    def __init__(self, data_flow: DataFlow):
        self._data_flow = data_flow

    def collect(self):
        return self._data_flow.get()

class AggregationFlowCollector(Collector):
    """AggregationFlowCollector is a collector aggregating multiple data flows"""
    def __init__(self, data_flows: List[DataFlow]):
        self._data_flows = data_flows
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._queue = Queue()
        self._stop_event = threading.Event()
        self._futures = self._executor.submit(self._collect)

    def collect(self):
        return self._queue.get()

    def _collect(self):
        logger.info(f"AggregationFlowCollector starts collecting data")
        while not self._stop_event.is_set():
            result = {}
            completed = []
            for data_flow in self._data_flows:
                data = data_flow.get()
                if isinstance(data, Error) or isinstance(data, Stop):
                    completed.append(data)
                    continue
                else:
                    result.update(data)
            self._queue.put(result)
            if all([isinstance(c, Stop) for c in completed]):
                self._queue.put(Stop("All data flows completed"))
            elif any([isinstance(c, Error) for c in completed]):
                self._queue.put(Error("One or more data flows ended in error"))

    def __del__(self):
        self._stop_event.set()
        self._executor.shutdown(wait=True)

class MultiFlowCollector(Collector):
    """MultiFlowCollector is a collector for multiple data flows"""
    def __init__(self, data_flows: List[DataFlow]):
        self._data_flows = data_flows
        self._executor = ThreadPoolExecutor(max_workers=len(data_flows))
        self._queue = Queue()
        self._stop_event = threading.Event()
        self._futures = [self._executor.submit(self._poll, i) for i in range(len(self._data_flows))]
        self._condition = threading.Condition()
        self._active_flow = -1

    def collect(self):
        return self._queue.get()

    def _poll(self, index: int):
        logger.info(f"Start polling data flow {self._data_flows[index].in_names}")
        data_flow = self._data_flows[index]
        while not self._stop_event.is_set():
            try:
                data = data_flow.get()
                with self._condition:
                    while self._active_flow != index and self._active_flow != -1:
                        logger.debug(f"Data flow {data_flow.in_names} is waiting for output to be free")
                        self._condition.wait()
                    if self._active_flow == -1:
                        self._active_flow = index
                if isinstance(data, Error) or isinstance(data, Stop):
                    logger.error(f"Data flow {data_flow.in_names} ended in: {data}")
                    self._queue.put(data)
                    self._active_flow = -1
                    self._condition.notify_all()
                else:
                    self._queue.put(data)
            except Empty:
                continue

    def __del__(self):
        self._stop_event.set()
        self._executor.shutdown(wait=True)

class ModelOperator:
    """An model operator runs a single model"""
    def __init__(self, model_config:Dict, model_repo: str):
        self._model_name = model_config["name"]
        self._model_home = os.path.join(model_repo, self._model_name)
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
    def model_config(self):
        return self._model_config.copy()

    @property
    def inputs(self):
        return self._in.copy()

    @property
    def outputs(self):
        return self._out.copy()

    def bind_input(self, configs: List[Dict], targets: List[str]=[]):
        if not targets:
            targets = [i['name'] for i in configs]
        flow = None
        tensor_names = [(i['name'], o) for i, o in zip(configs, targets)]
        tensor_types = [i['data_type'] for i in configs]
        image_tensor_type = None
        for tensor_type in tensor_types:
            if tensor_type in inbound_dataflow_mapping:
                image_tensor_type = tensor_type
                break
        if image_tensor_type is None:
            flow = DataFlow(configs, tensor_names, inbound=True)
        else:
            # customized inbound data flow
            flow = inbound_dataflow_mapping[image_tensor_type](configs, tensor_names, image_tensor_type)
        self._in.append(flow)
        logger.info(f"Data flow < {flow.in_names} -> {flow.o_names} > connected to model {self._model_name}")
        return flow

    def bind_output(self, configs: List[Dict], sources: List[str]=[]):
        if not sources:
            sources = [i['name'] for i in configs]
        tensor_names = [(i, o['name']) for i, o in zip(sources, configs)]
        flow = DataFlow(configs, tensor_names, outbound=True)
        self._out.append(flow)
        logger.info(f"model {self._model_name} connected to Data flow < {flow.in_names} -> {flow.o_names} >")
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
                                model_home=self._model_home
                            )
                        )
                    else:
                        raise Exception("Invalid Processor")
        # backend loop
        self._backend = model_backend
        collector = self._create_collector()
        while True:
            try:
                # collect input data until Stop is received
                data = collector.collect()
                if isinstance(data, Stop) or isinstance(data, Error):
                    # pass the error or stop message downstream
                    for out in self._out:
                        logger.debug(f"Passing error or stop message to {out.in_names}")
                        out.put(data)
                    continue
                # convert data to args and kwargs based on if explicit batching is required
                logger.debug(f"Input collected: {data}")
                args = []
                kwargs = data
                if any([isinstance(v, list) for _, v in kwargs.items()]):
                    # construct multiple inference requests
                    args = split_tensor_in_dict(kwargs)
                    kwargs = {}
                # call preprocess() before passing args to the backend
                processed = self._preprocess(args if args else [kwargs])
                if not processed:
                    logger.error(f"Empty result from preprocess: {args} and {kwargs}")
                    continue
                if args:
                    # explicitly batched data
                    args = processed
                elif kwargs:
                    # implicitly batched data
                    kwargs = processed[0]
                else:
                    logger.error(f"Invalid result from preprocess: {args} and {kwargs}")
                    continue
                # execute inference backend and collect result
                logger.debug(f"Model {self._model_name} invokes backend {self._backend.__class__.__name__} with {args if args else kwargs}")
                for r in self._backend(*args, **kwargs):
                    logger.debug(f"Model {self._model_name} generated result from backend {self._backend.__class__.__name__}: {r}")
                    if not self._out:
                        logger.error(f"No output data flow is bound to model {self._model_name}, please check the route configuration")
                        continue
                    if isinstance(r, Error):
                        logger.error(f"Error from model {self._model_name}: {r}")
                        continue
                    # iterate the result list and postprocess each of them
                    for out in self._out:
                        output_data = {n : [] for n in out.in_names}
                        if isinstance(r, list):
                            # we get a batch
                            for result in r:
                                result = self._postprocess(result)
                                if not all([n in result for n in out.in_names]):
                                    logger.error(f"Data received from model {self._model_name} is incomplete, expected: {out.in_names}, received: {result.keys()}. Post-processor missing?")
                                    continue
                                # collect the result
                                for n, v in output_data.items():
                                    if n in result:
                                        v.append(result[n])
                                    else:
                                        v.append(None)
                        else:
                            # implicit batching
                            output_data = self._postprocess(r)
                            if not all([n in output_data for n in out.in_names]):
                                logger.error(f"Data received from model {self._model_name} is incomplete, expected: {out.in_names}, received: {output_data.keys()}. Post-processor missing?")
                                continue
                        logger.debug(f"Deposit result: {output_data}")
                        out.put(output_data)
            except Exception as e:
                logger.exception(e)
                out.put(Error(str(e)))

    def _preprocess(self, args: List):
        # go through the preprocess chain
        outcome = args
        for preprocessor in self._preprocessors:
            result = []
            for data in outcome:
                # initialize the processed as the original values
                processed = {k: v for k, v in data.items()}
                # trigger the preprocessor if all the input tensors are present
                if all([i in data for i in preprocessor.input]):
                    input = [processed.pop(i) for i in preprocessor.input]
                    logger.debug(f"{self._model_name} invokes preprocessor {preprocessor.name} with given input {input}")
                    output = preprocessor(*input)
                    logger.debug(f"{self._model_name} preprocessor {preprocessor.name} generated output {output}")
                    if not isinstance(output, tuple):
                        logger.error("Return value of a processor must be a tuple")
                        continue
                    if len(output) != len(preprocessor.output):
                        logger.warning(f"Number of preprocessing output doesn't match the configuration, expecting {len(preprocessor.output)}, while getting {len(output)}")
                        continue
                    # update as processed
                    for key, value in zip(preprocessor.output, output):
                        processed[key] = value
                else:
                    logger.warning(f"Pre-processor {preprocessor.name} skipped because of missing input tensors")
                result.append(processed)
            # update outcome
            outcome = result
        # correct the data type
        for data in outcome:
            for key in data:
                value = data[key]
                i_config = next((i for i in self._model_config['input'] if i["name"] == key), None)
                if i_config is None:
                    logger.warning(f"Unexpected data: {key} from preprocessed")
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
                logger.warning(f"Post-processor {processor.name} skipped because of missing input tensors")
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

    def _create_collector(self):
        if len(self._in) == 1:
            logger.info(f"Single data flow input detected, using single flow collector on model {self._model_name}")
            return SingleFlowCollector(self._in[0])
        else:
            outputs = [set(d.o_names) for d in self._in]
            intersection = set.intersection(*outputs)
            if len(intersection) == 0:
                logger.info(f"Aggregation data flow input detected, using aggregation flow collector on model {self._model_name}")
                return AggregationFlowCollector(self._in)
            else:
                logger.info(f"Multi data flow input detected, using multi flow collector on model {self._model_name}")
                return MultiFlowCollector(self._in)

class InferenceBase:
    """The base model that drives the inference flow"""
    def initialize(self, model_repo: str):
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
        self._model_repo = model_repo

        if not os.path.exists(self._model_repo):
            logger.error(f"Model repository {self._model_repo} does not exist")
            raise Exception(f"Model repository {self._model_repo} does not exist")

        # set up the inference flow
        for model_config in global_config.models:
            self._operators.append(ModelOperator(OmegaConf.to_container(model_config), self._model_repo))

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
                        if len(s_configs) != len(route.data.source):
                            logger.error(f"Not all the sources are found in the input configs, unable to create passthrough dataflow")
                            continue
                        o_tensor_names = route.data.target if route.data.target else route.data.source
                        if any(n not in [c.name for c in global_config.output] for n in o_tensor_names):
                            logger.warning(f"Not all the output tensors from {o_tensor_names} are found in the output configs, consider adding a postprocessor to generate the missing tensors")
                        tensor_names = [(i['name'], o) for i, o in zip(s_configs, o_tensor_names)]
                        dataflow = DataFlow(configs=s_configs, tensor_names=tensor_names, inbound=True)
                    elif route.data.target:
                        t_configs = [OmegaConf.to_container(c) for c in global_config.output if c.name in route.data.target]
                        if len(t_configs) != len(route.data.target):
                            logger.error(f"Not all the targets are found in the output configs, unable to create passthrough dataflow")
                            continue
                        if any(n not in [c.name for c in global_config.input] for n in route.data.target):
                            logger.error(f"Not all the targets are found in the input configs, unable to create passthrough dataflow")
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
                            s_configs = []
                            for s in route.data.source:
                                c = next((c for c in global_config.input if c.name == s), None)
                                if c is None:
                                    logger.error(f"Input {s} not found in the input configs, unable to create dataflow")
                                    continue
                                s_configs.append(OmegaConf.to_container(c))
                        else:
                            s_configs = [OmegaConf.to_container(c) for c in global_config.input]
                        self._inputs.append(operator1.bind_input(s_configs, route.data.target))
                # only source model is specified.
                elif route.model.source:
                    # this is the top level output
                    operator = next((o for o in self._operators if o.model_name == route.model.source), None)
                    if operator is None:
                        logger.error(f"Model {route.model.source} in the routes not found")
                        continue
                    if route.data.target:
                        configs = [OmegaConf.to_container(c) for c in global_config.output if c.name in route.data.target]
                        self._outputs.append(operator.bind_output(configs, route.data.source))
                    else:
                        configs = [OmegaConf.to_container(c) for c in global_config.output if c.name in route.data.source]
                        if route.data.source == [c['name'] for c in configs]:
                            self._outputs.append(operator.bind_output(configs))
                        else:
                            logger.warning(f"Output of {operator.model_name} is not compatible with top level output, be sure to add a top level postprocessor")
                            dataflow = DataFlow(configs=None, tensor_names=[(i, i) for i in route.data.source])
                            operator.import_output(dataflow)
                            self._outputs.append(dataflow)
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
        if len(self._inputs) == 0 or len(self._outputs) == 0:
            error = "Either input or output is empty, inference pipeline is not complete"
            logger.error(error)
            raise Exception(error)
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