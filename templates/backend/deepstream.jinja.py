from pyservicemaker import Pipeline, Flow, BufferProvider, Buffer, RenderMode, BatchMetadataOperator, Probe, as_tensor, StateTransitionMessage
from typing import Dict, List
from queue import Queue, Empty, Full
from dataclasses import dataclass, field
import base64
import numpy as np
from abc import ABC, abstractmethod
import yaml
import tempfile
import os

png_data = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGK6HcwNCAAA//8DTgE8HuxwEQAAAABJRU5ErkJggg==")
jpg_data = base64.b64decode("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAgACADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD+f+iiigAooooAKKKKACiiigD/2Q==")


@dataclass
class PerfConfig:
    enable_fps_logs: bool = False
    enable_latency_logs: bool = False

@dataclass
class RenderConfig:
    enable_display: bool = False
    enable_osd: bool = False
    enable_stream: bool = False
    rtsp_mount_point: str | None = "/ds-test"
    rtsp_port: int | None = 8554

    def __bool__(self) -> bool:
        """Check if render configuration is valid."""
        if self.enable_display and self.enable_stream:
            logger.warning("RenderConfig: Both enable_display and enable_stream are True. both will be disabled.")
            return False
        if self.enable_osd and not self.enable_display and not self.enable_stream:
            logger.warning("RenderConfig: enable_osd is True but both enable_display and enable_stream are False. OSD requires an output method.")
            return False
        if self.enable_stream and (self.rtsp_mount_point is None or self.rtsp_port is None):
            logger.warning("RenderConfig: enable_stream is True but rtsp_mount_point or rtsp_port is None.")
            return False
        return True

@dataclass
class TrackerConfig:
    config_path: str | None = None
    lib_path: str | None = None

    def __bool__(self) -> bool:
        """Check if all required tracker configuration fields are set."""
        return self.config_path is not None and self.lib_path is not None

@dataclass
class MessageBrokerConfig:
    proto_lib_path: str | None = None
    msgconv_config_path: str | None = None
    conn_str: str | None = None
    topic: str | None = None

    def __bool__(self) -> bool:
        """Check if all required message broker configuration fields are set."""
        return (
            self.proto_lib_path is not None and
            self.msgconv_config_path is not None and
            self.conn_str is not None and
            self.topic is not None
        )

class ImageTensorInput(BufferProvider):

    def __init__(self, height, width, format):
        super().__init__()
        self.width = width
        self.height = height
        self.format = format
        self.framerate = 1
        self.device = 'cpu'
        self.queue = Queue(maxsize=1)

    def generate(self, size):
        tensor = self.queue.get()
        if isinstance(tensor, np.ndarray):
            return Buffer(tensor.tolist())
        elif isinstance(tensor, Stop):
            # EOS
            return Buffer()
        else:
            logger.exception("Unexpected input tensor data")
            return Buffer()

    def send(self, data):
        self.queue.put(data)

class GenericTensorInput():
    def __init__(self, device_id):
        self.queue = Queue(maxsize=1)
        self._device_id = device_id

    def generate(self, n):
        try:
            tensors = self.queue.get(timeout=1)
        except Empty:
            logger.warning("No tensor data to generate")
            return dict()
        result = {k: as_tensor(v, "").to_gpu(self._device_id) for k, v in tensors.items()}
        return result

    def send(self, data):
        self.queue.put(data)

class StaticTensorInput():
    def __init__(self, device_id):
        self._tensors = {}
        self._device_id = device_id

    def generate(self, n):
        result = {k: as_tensor(v, "").to_gpu(self._device_id) for k, v in self._tensors.items()}
        return result

    def set(self, data):
        self._tensors.update(data)

class TensorInputPool(ABC):
    @abstractmethod
    def submit(self, data: List):
        pass
    @abstractmethod
    def stop(self, reason: str):
        pass

class ImageTensorInputPool(TensorInputPool):

    def __init__(self, height, width, formats, batch_size, image_tensor_name, media_url_tensor_name, mime_tensor_name, device_id, require_extra_input):
        self._image_inputs = [ImageTensorInput(width, height, format) for format in formats for _ in range(batch_size)]
        self._media_url_tensor_name = media_url_tensor_name
        self._mime_tensor_name = mime_tensor_name
        self._image_tensor_name = image_tensor_name
        self._generic_input = GenericTensorInput(device_id) if require_extra_input else None
        self._batch_size = batch_size

    @property
    def image_inputs(self):
        return self._image_inputs

    @property
    def generic_input(self):
        return self._generic_input

    def submit(self, data: List):
        indices = []
        for item in data:
            mime_type = item.pop(self._mime_tensor_name, None)
            if mime_type is None:
                logger.error("MIME type is not specified")
                continue
            mime_type = mime_type.split('/');
            if mime_type[0] == 'image':
                format = mime_type[1].upper()
                # try find the free slot for the specific format
                i, image_tensor_input = next(((i, x) for i,x in enumerate(self._image_inputs) if x.format == format and x.queue.empty()), (-1, None))
                if image_tensor_input is None:
                    i, image_tensor_input = next(((i, x) for i,x in enumerate(self._image_inputs) if x.format == format), (-1, None))
                if image_tensor_input is not None:
                    if self._image_tensor_name in item:
                        image_tensor = item.pop(self._image_tensor_name)
                        image_tensor_input.send(image_tensor)
                        indices.append(i)
                    elif self._media_url_tensor_name in item:
                        image_url = item.pop(self._media_url_tensor_name)
                        with open(image_url, 'rb') as f:
                            image_tensor = np.frombuffer(f.read(), dtype=np.uint8)
                        image_tensor_input.send(image_tensor)
                        indices.append(i)
                    else:
                        logger.error(f"image tensor or media url is missing: {item}")
                else:
                    logger.error(f"Unable to find free slot for format {format}")
            else:
                logger.error(f"Unsupported MIME type {mime_type}")
                continue
        if self._generic_input:
            data = [data[i:i + self._batch_size] for i in range(0, len(data), self._batch_size)]
            for d in data:
                self._generic_input.send(stack_tensors_in_dict(d))
        # batched indices for each input
        return indices

    def stop(self, reason: str):
        for input in self._image_inputs:
            input.send(Stop(reason))
        if self._generic_input:
            self._generic_input.send(Stop(reason))

class BulkVideoInputPool(TensorInputPool):
    def __init__(self,
        max_batch_size,
        batch_timeout,
        media_url_tensor_name,
        source_tensor_name,
        mime_tensor_name,
        infer_config_paths,
        preprocess_config_paths,
        tracker_config: TrackerConfig,
        msgbroker_config: MessageBrokerConfig,
        render_config: RenderConfig,
        perf_config: PerfConfig,
        output,
        device_id,
        require_extra_input,
        engine_file_names,
        dims
    ):
        self._batch_size = max_batch_size
        self._batch_timeout = batch_timeout
        self._media_url_tensor_name = media_url_tensor_name
        self._source_tensor_name = source_tensor_name
        self._mime_tensor_name = mime_tensor_name
        self._infer_config_paths = infer_config_paths
        self._engine_file_names = engine_file_names
        self._preprocess_config_paths = preprocess_config_paths
        self._pipeline = None
        self._output = output
        self._generic_input = StaticTensorInput(device_id) if require_extra_input else None
        self._device_id = device_id
        self._dims = dims
        self._tracker_config = tracker_config
        self._msgbroker_config = msgbroker_config
        self._render_config = render_config
        self._perf_config = perf_config

    def submit(self, data: List):
        url_list = []
        source_config_file = None
        for item in data:
            if self._mime_tensor_name and self._mime_tensor_name in item:
                item.pop(self._mime_tensor_name)
            if self._media_url_tensor_name and self._media_url_tensor_name in item:
                url_list.append(item.pop(self._media_url_tensor_name))
            elif self._source_tensor_name and self._source_tensor_name in data[0]:
                source_config_file = item.pop(self._source_tensor_name).tolist()
                if not source_config_file.lower().endswith(('.yml', '.yaml')):
                    logger.error(
                        f"Source config file must be a YAML file: {source_config_file}"
                    )
                    return []
                break
            else:
                logger.error(f"Invalid input data: {data}")
                continue

        if len(url_list) > self._batch_size:
            logger.warning(
                f"Number of media urls ({len(url_list)}) > "
                f"batch size ({self._batch_size}), "
                f"only the first {self._batch_size} will be used"
            )
            url_list = url_list[:self._batch_size]

        pipeline = Pipeline(f"deepstream-video-batch")
        if self._generic_input and data:
            self._generic_input.set(stack_tensors_in_dict(data))

        # Use appropriate batch_capture method based on input type
        if source_config_file:
            flow = Flow(pipeline).batch_capture(
                input=source_config_file,
                width=self._dims[1], height=self._dims[0],
                batch_size=self._batch_size
            )
        else:
            flow = Flow(pipeline).batch_capture(
                url_list,
                width=self._dims[1],
                height=self._dims[0],
                batched_push_timeout=self._batch_timeout)

        for config in self._preprocess_config_paths:
            flow = flow.preprocess(config, None if not self._generic_input else self._generic_input.generate)
        for config_path, engine_file in zip(self._infer_config_paths, self._engine_file_names):
            if engine_file:
                flow = flow.infer(config_path, batch_size=self._batch_size, model_engine_file=engine_file)
            else:
                flow = flow.infer(config_path, batch_size=self._batch_size)
        if self._tracker_config:
            flow = flow.track(
                ll_config_file=self._tracker_config.config_path,
                ll_lib_file=self._tracker_config.lib_path,
                gpu_id=self._device_id,
                tracker_width=self._dims[1],
                tracker_height=self._dims[0]
            )
        flow = flow.attach(Probe('tensor_retriver', self._output))

        if self._perf_config.enable_fps_logs:
            flow = flow.attach(what="measure_fps_probe", name="fps_probe")
        if self._perf_config.enable_latency_logs:
            flow = flow.attach(what="measure_latency_probe", name="latency_probe")

        if self._msgbroker_config:
            flow = flow.attach(
                what="add_message_meta_probe",
                name="message_generator"
            )

            flow = flow.fork()
            flow.publish(
                msg_broker_proto_lib=self._msgbroker_config.proto_lib_path,
                msg_broker_conn_str=self._msgbroker_config.conn_str,
                topic=self._msgbroker_config.topic,
                msg_conv_config=self._msgbroker_config.msgconv_config_path,
                sync=False
            )
        if self._render_config.enable_stream:
            flow.render(RenderMode.STREAM,
                       enable_osd=self._render_config.enable_osd,
                       rtsp_mount_point=self._render_config.rtsp_mount_point,
                       rtsp_port=self._render_config.rtsp_port,
                       sync=False)
        else:
            flow.render(RenderMode.DISCARD if not self._render_config.enable_display else RenderMode.DISPLAY,
                       enable_osd=self._render_config.enable_osd, sync=False)

        if self._pipeline is not None:
            self._pipeline.wait()

        pipeline.start()
        self._pipeline = pipeline
        return list(range(len(url_list))) if url_list else list(range(self._batch_size))

    def stop(self, reason: str):
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline.wait()


class BaseTensorOutput(BatchMetadataOperator):
    def __init__(self, n_outputs, name: str = None):
        super().__init__()
        self._queue = Queue(maxsize=n_outputs)
        self._stashed = None
        self._name = name

    def handle_metadata(self, batch_meta):
        pass

    def collect(self, indices: List, timeout=None) -> List | None:
        # expected results for a batch
        collected = [None] * (max(indices) + 1)

        # first check the for stashed results for the batch
        if self._stashed is not None:
            collected[self._stashed[0]] = self._stashed[1]
            self._stashed = None

        # read from the thread queue
        while not all(collected[i] is not None for i in indices):
            try:
                i, data = self._queue.get(timeout=timeout)
                if collected[i] is None:
                    collected[i] = data
                else:
                    # this is a new batch, stash the results and break
                    self._stashed = (i, data)
                    break
            except Empty:
                break
        if all(i is None for i in collected):
            # signal the end of the inference run
            return None
        # rearrange the results to the required order
        collected = [collected[i] for i in indices]
        return collected if self._name is None else [
            {self._name: r} for r in collected
        ]

    def reset(self):
        self._queue = Queue(maxsize=self._queue.maxsize)
        self._stashed = None

    def _deposit(self, index: int, data: dict):
        logger.debug(
            f"DeepstreamBackend: Depositing data to index {index}: {data}"
        )
        try:
            self._queue.put((index, data))
        except Full:
            logger.warning(
                f"DeepstreamBackend: Queue is full, dropping data from "
                f"index {index}"
            )


class TensorOutput(BaseTensorOutput):
    def __init__(self, n_outputs, preprocess_config_path):
        super().__init__(n_outputs, None)
        self._preprocess_config_path = preprocess_config_path

    def handle_metadata(self, batch_meta):
        if self._preprocess_config_path:
            for meta in batch_meta.preprocess_batch_items:
                preprocess_batch = meta.as_preprocess_batch()
                if not preprocess_batch:
                    continue
                for roi in preprocess_batch.rois:
                    result = dict()
                    for user_meta in roi.tensor_items:
                        tensor_output = user_meta.as_tensor_output()
                        if tensor_output :
                            for n, tensor in tensor_output.get_layers().items():
                                torch_tensor = torch.utils.dlpack.from_dlpack(tensor).to('cpu')
                                result[n] = torch_tensor
                    self._deposit(roi.frame_meta.pad_index, result)
        else:
            for frame_meta in batch_meta.frame_items:
                result = dict()
                for user_meta in frame_meta.tensor_items:
                    tensor_output = user_meta.as_tensor_output()
                    if tensor_output :
                        for n, tensor in tensor_output.get_layers().items():
                            torch_tensor = torch.utils.dlpack.from_dlpack(tensor).to('cpu')
                            result[n] = torch_tensor
                self._deposit(frame_meta.pad_index, result)

@dataclass
class DeepstreamMetadata:
    shape: list[int] = field(default_factory=lambda: [0, 0])
    bboxes: list[list[int]] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    seg_maps: list[list[int]] = field(default_factory=list)
    objects: list[int] = field(default_factory=list)
    timestamp: int = 0

def _to_list(array):
    """Adapt data to python list[int], mainly for NumPy array to list[int]"""
    if hasattr(array, 'tolist'):
        # Convert NumPy array to list and flatten to 1D
        return array.flatten().tolist()
    elif isinstance(array, list):
        # Already a list, return as is
        return array
    else:
        # Fallback: try to convert to list
        return list(array)

class PreprocessMetadataOutput(BaseTensorOutput):
    def __init__(self, n_outputs, output_name, dims):
        super().__init__(n_outputs, name=output_name)
        self._shape = dims

    def handle_metadata(self, batch_meta):
        for meta in batch_meta.preprocess_batch_items:
            preprocess_batch = meta.as_preprocess_batch()
            if not preprocess_batch:
                continue
            for roi in preprocess_batch.rois:
                metadata = DeepstreamMetadata()
                # segmentation metadata
                for u_meta in roi.segmentation_items:
                    seg_meta = u_meta.as_segmentation()
                    if seg_meta:
                        metadata.shape = [seg_meta.height, seg_meta.width]
                        metadata.seg_maps.append(_to_list(seg_meta.class_map))
                # object metadata
                for object_meta in roi.frame_meta.object_items:
                    labels = [object_meta.label] if object_meta.label else []
                    left = int(object_meta.rect_params.left)
                    top = int(object_meta.rect_params.top)
                    width = int(object_meta.rect_params.width)
                    height = int(object_meta.rect_params.height)
                    metadata.shape = [self._shape[0], self._shape[1]]
                    metadata.bboxes.append([left, top, left + width, top + height])
                    metadata.probs.append(object_meta.confidence)
                    for classifier in object_meta.classifier_items:
                        for i in range(classifier.n_labels):
                            labels.append(classifier.get_n_label(i))
                    metadata.labels.append(labels)
                    metadata.objects.append(object_meta.object_id)
                for classifier in roi.classifier_items:
                    labels = []
                    for i in range(classifier.n_labels):
                        labels.append(classifier.get_n_label(i))
                    metadata.labels.append(labels)
                metadata.timestamp = roi.frame_meta.buffer_pts
                self._deposit(roi.frame_meta.pad_index, metadata)

class MetadataOutput(BaseTensorOutput):
    def __init__(self, n_outputs, output_name, dims):
        super().__init__(n_outputs, name=output_name)
        self._shape = dims

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            metadata = DeepstreamMetadata()
            for object_meta in frame_meta.object_items:
                labels = [object_meta.label] if object_meta.label else []
                left = int(object_meta.rect_params.left)
                top = int(object_meta.rect_params.top)
                width = int(object_meta.rect_params.width)
                height = int(object_meta.rect_params.height)
                metadata.shape = [self._shape[0], self._shape[1]]
                metadata.bboxes.append([left, top, left + width, top + height])
                metadata.probs.append(object_meta.confidence)
                for classifier in object_meta.classifier_items:
                    for i in range(classifier.n_labels):
                        labels.append(classifier.get_n_label(i))
                metadata.labels.append(labels)
                metadata.objects.append(object_meta.object_id)
            for user_meta in frame_meta.segmentation_items:
                seg_meta = user_meta.as_segmentation()
                if seg_meta:
                    metadata.shape = [seg_meta.height, seg_meta.width]
                    metadata.seg_maps.append(_to_list(seg_meta.class_map))
            metadata.timestamp = frame_meta.buffer_pts
            self._deposit(frame_meta.pad_index, metadata)

class DeepstreamBackend(ModelBackend):
    """Deepstream backend using pyservicemaker"""
    def __init__(self, model_config:Dict, model_home: str, device_id: int=0):
        super().__init__(model_config, model_home, device_id)
        self._max_batch_size = model_config["max_batch_size"]
        self._model_name = model_config["name"]
        self._output_names = [o['name'] for o in model_config['output']]
        self._output_types = [o['data_type'] for o in model_config['output']]
        self._image_tensor_name = None
        self._media_url_tensor_name = None
        self._mime_tensor_name = None
        self._source_tensor_name = None
        self._inference_timeout = model_config["parameters"].get("inference_timeout", 3)

        if len(self._output_names) > 1 and self._output_types[0] == "TYPE_CUSTOM_DS_METADATA":
            raise Exception(f"No more than one output is allowed for DS metadata!")
        tensor_output = False if self._output_types[0] == "TYPE_CUSTOM_DS_METADATA" else True
        dims = (0, 0)
        if "parameters" not in model_config or "infer_config_path" not in model_config["parameters"]:
            raise Exception("Deepstream pipeline requires infer_config_path")
        infer_config_paths = self._correct_config_paths(model_config["parameters"]['infer_config_path'])
        if not infer_config_paths:
            raise Exception("Deepstream pipeline requires infer_config_path")
        preprocess_config_paths = []
        if "preprocess_config_path" in model_config["parameters"]:
            preprocess_config_paths = self._correct_config_paths(model_config["parameters"]['preprocess_config_path'])
        if "tracker_config" in model_config["parameters"]:
            tracker_config = TrackerConfig(
                config_path=self._correct_config_paths(
                    [model_config["parameters"]["tracker_config"].get("ll_config_file")]
                )[0] if model_config["parameters"]["tracker_config"].get("ll_config_file") else None,
                lib_path=self._correct_config_paths(
                    [model_config["parameters"]["tracker_config"]["ll_lib_file"]]
                )[0]
            )
            if not tracker_config:
                logger.warning("DeepstreamBackend: tracker_config is not properlyconfigured")
        else:
            tracker_config = TrackerConfig()
        if "msgbroker_config" in model_config["parameters"]:
            msgbroker_config = MessageBrokerConfig(
                proto_lib_path=self._correct_config_paths(
                    [model_config["parameters"]["msgbroker_config"]["msgbroker_proto_lib_path"]]
                )[0],
                msgconv_config_path=self._correct_config_paths(
                    [model_config["parameters"]["msgbroker_config"]["msgbroker_msgconv_config_path"]]
                )[0],
                conn_str=model_config["parameters"]["msgbroker_config"]["msgbroker_conn_str"],
                topic=model_config["parameters"]["msgbroker_config"]["msgbroker_topic"]
            )
            if not msgbroker_config:
                logger.warning("DeepstreamBackend: msgbroker_config is not properly configured")
        else:
            msgbroker_config = MessageBrokerConfig()
        if "render_config" in model_config["parameters"]:
            render_config = RenderConfig(
                enable_display=model_config["parameters"]["render_config"].get("enable_display", False),
                enable_osd=model_config["parameters"]["render_config"].get("enable_osd", False),
                enable_stream=model_config["parameters"]["render_config"].get("enable_stream", False),
                rtsp_mount_point=model_config["parameters"]["render_config"].get("rtsp_mount_point", "/ds-test"),
                rtsp_port=model_config["parameters"]["render_config"].get("rtsp_port", 8554)
            )
        else:
            render_config = RenderConfig()
        if "perf_config" in model_config["parameters"]:
            perf_config = PerfConfig(
                enable_fps_logs=model_config["parameters"]["perf_config"].get("enable_fps_logs", False),
                enable_latency_logs=model_config["parameters"]["perf_config"].get("enable_latency_logs", False)
            )
        else:
            perf_config = PerfConfig()
        infer_element = model_config['backend'].split('/')[-1]
        with_triton = infer_element == 'nvinferserver'
        require_extra_input = False
        warmup_data_0 = dict()
        warmup_data_1 = dict()
        for input in model_config['input']:
            if input['data_type'] == 'TYPE_CUSTOM_DS_IMAGE':
                self._image_tensor_name = input['name']
            elif input['data_type'] == 'TYPE_CUSTOM_BINARY_URLS':
                self._media_url_tensor_name = input['name']
            elif input['data_type'] == 'TYPE_CUSTOM_DS_MIME':
                self._mime_tensor_name = input['name']
            elif input['data_type'] == 'TYPE_CUSTOM_DS_SOURCE_CONFIG':
                self._source_tensor_name = input['name']
            elif not ('optional' in input and input['optional']):
                tensor_name = input['name']
                require_extra_input = True
                np_type = np_datatype_mapping[input['data_type']]
                warmup_data_0[tensor_name] = np.random.rand(*input['dims']).astype(np_type)
                warmup_data_1[tensor_name] = np.random.rand(*input['dims']).astype(np_type)
        if (self._image_tensor_name is None and
            self._media_url_tensor_name is None and
            self._source_tensor_name is None):
            raise ValueError(
                "Deepstream pipeline requires at least one "
                "TYPE_CUSTOM_DS_IMAGE or TYPE_CUSTOM_BINARY_URLS input "
                "or TYPE_CUSTOM_DS_SOURCE_CONFIG input"
            )
        if ((self._image_tensor_name or self._media_url_tensor_name) and
            self._mime_tensor_name is None):
            raise ValueError("Deepstream pipeline requires TYPE_CUSTOM_DS_MIME input")
        # override the network dimensions from  the primary inference config
        try:
            primary_infer_config_path = infer_config_paths[0]
            with open(primary_infer_config_path, 'r') as f:
                primary_infer_config = yaml.safe_load(f)
            if "property" in primary_infer_config:
                property = primary_infer_config["property"]
                if "infer-dims" in property:
                    infer_dims = [int(dim) for dim in property["infer-dims"].split(";")]
                    if len(infer_dims) == 3:
                        dims = (infer_dims[0], infer_dims[1]) if "network-input-order" in property and property["network-input-order"] == 1 else (infer_dims[1], infer_dims[2])
                        logger.info(f"DeepstreamBackend: setting network dimensions to {dims}")
        except Exception as e:
            raise RuntimeError(f"Failed to load primary inference config: {e}") from e
        if dims[0] == 0 or dims[1] == 0:
            raise ValueError(
                "DeepstreamBackend: unable to find network dimensions: "
                "infer-dims missing in the config?"
            )
        # construct the input pools, outputs and pipelines
        self._in_pools = {}
        self._outputs = {}
        self._pipelines = {}
        if self._image_tensor_name is not None or self._media_url_tensor_name is not None:
            # image input support
            media = "image"
            formats = ["JPEG", "PNG"]
            in_pool = ImageTensorInputPool(dims[0], dims[1], formats, self._max_batch_size, self._image_tensor_name, self._media_url_tensor_name, self._mime_tensor_name, device_id, require_extra_input)
            n_output = self._max_batch_size * len(formats)
            if tensor_output:
                output = TensorOutput(n_output, preprocess_config_paths)
            elif preprocess_config_paths:
                output = PreprocessMetadataOutput(n_output, self._output_names[0], dims)
            else:
                output = MetadataOutput(n_output, self._output_names[0], dims)
            # create the pipeline
            pipeline = Pipeline(f"deepstream-{self._model_name}-{media}")

            self._pipelines[media] = pipeline
            self._in_pools[media] = in_pool
            self._outputs[media] = output

            # build the inference flow
            flow = Flow(pipeline)
            probe = Probe('tensor_retriver', output)
            batch_timeout = model_config["parameters"].get("batch_timeout", 1000 * self._max_batch_size)
            flow = flow.inject(in_pool.image_inputs).decode().batch(batch_size=self._max_batch_size, batched_push_timeout=batch_timeout, live_source=False, width=dims[1], height=dims[0])
            for config_file in preprocess_config_paths:
                input = self._in_pools[media].generic_input if require_extra_input else None
                flow = flow.preprocess(config_file, None if not input else input.generate)
            for config_file in infer_config_paths:
                engine_file = self._generate_engine_name(config_file, device_id, self._max_batch_size)
                if engine_file:
                    flow = flow.infer(config_file, with_triton, batch_size=self._max_batch_size, model_engine_file=engine_file)
                else:
                    flow = flow.infer(config_file, with_triton, batch_size=self._max_batch_size)
            flow = flow.attach(probe).render(RenderMode.DISCARD, enable_osd=False)
            pipeline.start()
            # warm up
            warmup_data_0[self._image_tensor_name] = np.frombuffer(png_data, dtype=np.uint8)
            warmup_data_0[self._mime_tensor_name] = "image/png"
            warmup_data_1[self._image_tensor_name] = np.frombuffer(jpg_data, dtype=np.uint8)
            warmup_data_1[self._mime_tensor_name] = "image/jpeg"
            indices = in_pool.submit([warmup_data_0.copy() for _ in range(self._max_batch_size)])
            results = output.collect(indices)
            output.reset()
            logger.info(f"Warm up 0: {results}")
            indices = in_pool.submit([warmup_data_1.copy() for _ in range(self._max_batch_size)])
            results = output.collect(indices)
            output.reset()
            logger.info(f"Warm up 1: {results}")

        if (self._media_url_tensor_name is not None or
            self._source_tensor_name is not None):
            # video input support
            media = "video"
            if tensor_output:
                output = TensorOutput(self._max_batch_size, preprocess_config_paths)
            elif preprocess_config_paths:
                output = PreprocessMetadataOutput(self._max_batch_size, self._output_names[0], dims)
            else:
                output = MetadataOutput(self._max_batch_size, self._output_names[0], dims)
            engine_files = [
                self._generate_engine_name(
                    config_file,
                    device_id,
                    self._max_batch_size
                )
                for config_file in infer_config_paths
            ]
            in_pool = BulkVideoInputPool(
                max_batch_size=self._max_batch_size,
                batch_timeout=model_config["parameters"].get("batch_timeout", 1000 * self._max_batch_size),
                media_url_tensor_name=self._media_url_tensor_name,
                source_tensor_name=self._source_tensor_name,
                mime_tensor_name=self._mime_tensor_name,
                infer_config_paths=infer_config_paths,
                preprocess_config_paths=preprocess_config_paths,
                tracker_config=tracker_config,
                msgbroker_config=msgbroker_config,
                render_config=render_config,
                perf_config=perf_config,
                output=output,
                device_id=device_id,
                require_extra_input=require_extra_input,
                engine_file_names=engine_files,
                dims=dims,
            )
            if not all(os.path.exists(e) for e in engine_files):
                # generate the engine files
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Write test data to a temporary file
                    temp_data_path = os.path.join(temp_dir, 'test.png')
                    with open(temp_data_path, 'wb') as f:
                        f.write(png_data)
                    warmup_data = {self._media_url_tensor_name: temp_data_path}
                    indices = in_pool.submit([warmup_data])
                    results = output.collect(indices)
                    output.reset()

        self._in_pools[media] = in_pool
        self._outputs[media] = output

        logger.info(
            f"DeepstreamBackend created for {self._model_name} to generate "
            f"{self._output_names}, output tensor: {tensor_output}"
        )


    def __call__(self, *args, **kwargs):
        in_data_list = args if args else [kwargs]
        media = None
        explicit_batch = True if args else False

        # analyze the input batch
        for data in in_data_list:
            # get the media type
            if ((self._image_tensor_name in data or self._media_url_tensor_name in data) and
                not self._mime_tensor_name in data):
                logger.error(f"MIME type is not specified for input {data}")
                return
            if self._source_tensor_name and self._source_tensor_name in data:
                media = "video"
                explicit_batch = True
                break # only video source is supported through source config

            current_media = data[self._mime_tensor_name].split('/')[0]
            if media is None:
                media = current_media
            elif media != current_media:
                logger.error(
                    f"Mixed media types are not supported in a single batch, "
                    f"got {media} and {current_media}"
                )
                return

        # submit the data to the pipeline which supports the media type
        indices = self._in_pools[media].submit(in_data_list)
        # collect the results
        while True:
            # TODO: timeout should be runtime configurabl
            results = self._outputs[media].collect(indices, timeout=self._inference_timeout)
            if not results:
                logger.info("DeepstreamBackend: No more data from this batch")
                break
            out_data_list = []
            for result in results:
                out_data = dict()
                for o in self._output_names:
                    if o in result:
                        out_data[o] = result[o]
                    else:
                        out_data[o] = None
                out_data_list.append(out_data)
            yield out_data_list if explicit_batch else out_data_list[0]
            # No consecutive inference results for image
            if media == "image":
                break

    def __del__(self):
        for input in self._in_pools.values():
            input.stop("Finalized")
        for pipeline in self._pipelines.values():
            pipeline.stop()
            pipeline.wait()

    def _generate_engine_name(self, config_path: str, device_id: int, batch_size: int):
        def network_mode_to_string(network_mode: int):
            if network_mode == 0:
                return "fp32"
            elif network_mode == 1:
                return "int8"
            elif network_mode == 2:
                return "fp16"
            else:
                return ""
        network_mode = "fp16"
        onnx_file = "model.onnx"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if not "property" in config:
                return None
            property = config["property"]
            if not "onnx-file" in property:
                return None
            onnx_file = property["onnx-file"]
            if "network-mode" in property:
                mode = network_mode_to_string(property["network-mode"])
                if mode:
                    network_mode = mode
        engine_file = f"{onnx_file}_b{batch_size}_gpu{device_id}_{network_mode}.engine"
        return os.path.join(self._model_home, engine_file)

    def _correct_config_paths(self, config_paths: List[str]) -> List[str]:
        if not config_paths:
            return []
        for i, path in enumerate(config_paths):
            if not os.path.isabs(path):
                config_paths[i] = os.path.join(self._model_home, path)
        return config_paths
