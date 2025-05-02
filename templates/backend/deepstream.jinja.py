from pyservicemaker import Pipeline, Flow, BufferProvider, Buffer, RenderMode, BatchMetadataOperator, Probe, as_tensor, StateTransitionMessage
from typing import Dict, List
from queue import Queue, Empty
from dataclasses import dataclass, field
import base64
import numpy as np
from abc import ABC, abstractmethod
import yaml

png_data = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGK6HcwNCAAA//8DTgE8HuxwEQAAAABJRU5ErkJggg==")
jpg_data = base64.b64decode("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAgACADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD+f+iiigAooooAKKKKACiiigD/2Q==")


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

    def generate(self):
        try:
            tensors = self.queue.get(timeout=1)
        except Empty:
            logger.warning("No tensor data to generate")
            return dict()
        result = {k: as_tensor(v, "").to_gpu(self._device_id) for k, v in tensors.items()}
        return result

    def send(self, data):
        self.queue.put(data)

class TensorInputPool(ABC):
    @abstractmethod
    def submit(self, data: List):
        pass
    @abstractmethod
    def stop(self):
        pass

class ImageTensorInputPool(TensorInputPool):

    def __init__(self, height, width, formats, batch_size, image_tensor_name, media_url_tensor_name, mime_tensor_name, device_id, require_extra_input=False):
        self._image_inputs = [ImageTensorInput(width, height, format) for format in formats for _ in range(batch_size)]
        self._media_url_tensor_name = media_url_tensor_name
        self._mime_tensor_name = mime_tensor_name
        self._image_tensor_name = image_tensor_name
        self._generic_input = GenericTensorInput(device_id) if require_extra_input else None

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
                        logger.error(f"Unable to find tensor input for format {format}")
                else:
                    logger.error(f"Unable to find free slot for format {format}")
            else:
                logger.error(f"Unsupported MIME type {mime_type}")
                continue
        if self._generic_input:
            self._generic_input.send(stack_tensors_in_dict(data))
        # batched indices for each input
        return indices

    def stop(self):
        for input in self._image_inputs:
            input.send(Stop())
        if self._generic_input:
            self._generic_input.send(Stop())

class BulkVideoInputPool(TensorInputPool):
    def __init__(self, media_url_tensor_name, mime_tensor_name, infer_config_path, output):
        self._media_url_tensor_name = media_url_tensor_name
        self._mime_tensor_name = mime_tensor_name
        self._infer_config_path = infer_config_path
        self._pipeline = None
        self._output = output

    def submit(self, data: List):
        try:
            url_list = [item.pop(self._media_url_tensor_name) for item in data]
        except KeyError:
            logger.error(f"Unable to find tensor input for media_url_tensor_name {self._media_url_tensor_name}")
            return []

        pipeline = Pipeline(f"deepstream-video-batch")
        flow = Flow(pipeline).batch_capture(url_list)
        for config_path in self._infer_config_path:
            flow = flow.infer(config_path)
        flow = flow.attach(Probe('tensor_retriver', self._output)).render(RenderMode.DISCARD, enable_osd=False)

        if self._pipeline is not None:
            self._pipeline.wait()
        logger.info("DeepstreamBackend: starting pipeline for bulk video inference...")
        pipeline.start()
        self._pipeline = pipeline
        return [i for i in range(len(url_list))]

    def stop(self):
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline.join()

class BaseTensorOutput(BatchMetadataOperator):
    def __init__(self, n_outputs, name: str = None):
        super().__init__()
        self._queues = [Queue() for _ in range(n_outputs)]
        self._name = name

    def handle_metadata(self, batch_meta):
        pass

    def collect(self, indices: List, timeout=None) -> List | None:
        try:
            results = [self._queues[i].get(timeout=timeout) if i >= 0 else None for i in indices]
        except Empty:
            return None
        if any (x is None for x in results):
            return None
        return results if self._name is None else [{self._name: r} for r in results]

    def reset(self):
        self._queues = [Queue() for _ in range(len(self._queues))]

    def _deposit(self, index: int, data: dict):
        logger.debug(f"DeepstreamBackend: Depositing data to index {index}: {data}")
        q = self._queues[index]
        q.put(data)


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
    timestamp: int = 0

class PreprocessMetadataOutput(BaseTensorOutput):
    def __init__(self, n_outputs, output_name, d):
        super().__init__(n_outputs, name=output_name)
        self._shape = d

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
                        metadata.seg_maps.append(seg_meta.class_map)
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
                for classifier in roi.classifier_items:
                    labels = []
                    for i in range(classifier.n_labels):
                        labels.append(classifier.get_n_label(i))
                    metadata.labels.append(labels)
                metadata.timestamp = roi.frame_meta.buffer_pts
                self._deposit(roi.frame_meta.pad_index, {"data": metadata})

class MetadataOutput(BaseTensorOutput):
    def __init__(self, n_outputs, output_name, d):
        super().__init__(n_outputs, name=output_name)
        self._shape = d

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
            for user_meta in frame_meta.segmentation_items:
                seg_meta = user_meta.as_segmentation()
                if seg_meta:
                    metadata.shape = [seg_meta.height, seg_meta.width]
                    metadata.seg_maps.append(seg_meta.class_map)
            metadata.timestamp = frame_meta.buffer_pts
            self._deposit(frame_meta.pad_index, {"data": metadata})

class DeepstreamBackend(ModelBackend):
    """Deepstream backend using pyservicemaker"""
    def __init__(self, model_config:Dict, model_home: str, device_id: int=0):
        super().__init__(model_config, model_home, device_id)
        self._max_batch_size = model_config["max_batch_size"]
        self._model_name = model_config["name"]
        self._output_names = [o['name'] for o in model_config['output']]
        self._output_types = [o['data_type'] for o in model_config['output']]
        self._pass_through_tensors = []
        self._image_tensor_name = None
        self._media_url_tensor_name = None
        self._mime_tensor_name = None

        if len(self._output_names) > 1 and self._output_types[0] == "TYPE_CUSTOM_DS_METADATA":
            raise Exception(f"No more than one output is allowed for DS metadata!")
        tensor_output = False if self._output_types[0] == "TYPE_CUSTOM_DS_METADATA" else True
        dims = model_config['input'][0]['dims']
        d = (dims[1], dims[2]) if dims[0] == 3 else (dims[0], dims[1])
        if "parameters" not in model_config or "infer_config_path" not in model_config["parameters"]:
            raise Exception("Deepstream pipeline requires infer_config_path")
        infer_config_path = model_config["parameters"]['infer_config_path']
        if not infer_config_path:
            raise Exception("Deepstream pipeline requires infer_config_path")
        preprocess_config_path = model_config["parameters"]['preprocess_config_path'] if "preprocess_config_path" in model_config["parameters"] else []
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
            elif input['data_type'] == 'TYPE_CUSTOM_DS_PASSTHROUGH':
                self._pass_through_tensors.append(input['name'])
            elif input['data_type'] == 'TYPE_CUSTOM_DS_MIME':
                self._mime_tensor_name = input['name']
            elif not ('optional' in input and input['optional']):
                tensor_name = input['name']
                require_extra_input = True
                np_type = np_datatype_mapping[input['data_type']]
                warmup_data_0[tensor_name] = np.random.rand(*input['dims']).astype(np_type)
                warmup_data_1[tensor_name] = np.random.rand(*input['dims']).astype(np_type)
        if self._image_tensor_name is None and self._media_url_tensor_name is None:
            raise Exception("Deepstream pipeline requires at least one TYPE_CUSTOM_DS_IMAGE or TYPE_CUSTOM_BINARY_URLS input")
        if self._mime_tensor_name is None:
            raise Exception("Deepstream pipeline requires at least one TYPE_CUSTOM_DS_MIME input")
        # override the network dimensions from  the primary inference config
        try:
            primary_infer_config_path = infer_config_path[0]
            if not os.path.isabs(primary_infer_config_path):
                primary_infer_config_path = os.path.join(self._model_home, primary_infer_config_path)
            with open(primary_infer_config_path, 'r') as f:
                primary_infer_config = yaml.safe_load(f)
            if "property" in primary_infer_config:
                property = primary_infer_config["property"]
                if "infer-dims" in property:
                    infer_dims = [int(dim) for dim in property["infer-dims"].split(";")]
                    if len(infer_dims) == 3:
                        d = (infer_dims[0], infer_dims[1]) if "network-input-order" in property and property["network-input-order"] == 1 else (infer_dims[1], infer_dims[2])
                        logger.info(f"DeepstreamBackend: overriding network dimensions to {d}")
        except Exception as e:
            raise Exception(f"Failed to load primary inference config: {e}")
        # construct the input pools, outputs and pipelines
        self._in_pools = {}
        self._outputs = {}
        self._pipelines = {}
        if self._image_tensor_name is not None:
            # image input support
            media = "image"
            formats = ["JPEG", "PNG"]
            in_pool = ImageTensorInputPool(d[0], d[1], formats, self._max_batch_size, self._image_tensor_name, self._media_url_tensor_name, self._mime_tensor_name, device_id, require_extra_input)
            n_output = self._max_batch_size * len(formats)
            if tensor_output:
                output = TensorOutput(n_output, preprocess_config_path)
            elif preprocess_config_path:
                output = PreprocessMetadataOutput(n_output, self._output_names[0], d)
            else:
                output = MetadataOutput(n_output, self._output_names[0], d)
            # create the pipeline
            pipeline = Pipeline(f"deepstream-{self._model_name}-{media}")

            self._pipelines[media] = pipeline
            self._in_pools[media] = in_pool
            self._outputs[media] = output

            # build the inference flow
            flow = Flow(pipeline)
            probe = Probe('tensor_retriver', output)
            batch_timeout = 1000 * self._max_batch_size
            flow = flow.inject(in_pool.image_inputs).decode().batch(batch_size=self._max_batch_size, batched_push_timeout=batch_timeout, live_source=False, width=d[1], height=d[0])
            for config in preprocess_config_path:
                config_file = config
                if not os.path.isabs(config):
                    config_file = os.path.join(self._model_home, config)
                flow = flow.preprocess(config_file, None if not require_extra_input else lambda: self._in_pools[media].generic_input.generate())
            for config in infer_config_path:
                config_file = config
                if not os.path.isabs(config):
                    config_file = os.path.join(self._model_home, config)
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

        if self._media_url_tensor_name is not None:
            # video input support
            media = "video"
            if tensor_output:
                output = TensorOutput(self._max_batch_size, preprocess_config_path)
            elif preprocess_config_path:
                output = PreprocessMetadataOutput(self._max_batch_size, self._output_names[0], d)
            else:
                output = MetadataOutput(self._max_batch_size, self._output_names[0], d)
            in_pool = BulkVideoInputPool(self._media_url_tensor_name, self._mime_tensor_name, infer_config_path, output)

            self._in_pools[media] = in_pool
            self._outputs[media] = output

        logger.info(f"DeepstreamBackend created for {self._model_name} to generate {self._output_names}, output tensor: {tensor_output}")


    def __call__(self, *args, **kwargs):
        logger.debug(f"DeepstreamBackend {self._model_name} triggerred with  {args if args else kwargs}")
        in_data_list = args if args else [kwargs]
        media = None
        pass_through_list = [dict() for _ in range(len(in_data_list))]

        # analyze the input batch
        for data, pass_through_data in zip(in_data_list, pass_through_list):
            # pass through the tensors if needed
            for tensor_name in self._pass_through_tensors:
                pass_through_data[tensor_name] = data.pop(tensor_name, None)
            # get the media type
            if not self._mime_tensor_name in data:
                raise Exception(f"MIME type is not specified for input {data}")
            current_media = data[self._mime_tensor_name].split('/')[0]
            if media is None:
                media = current_media
            elif media != current_media:
                raise Exception(f"Mixed media types are not supported in a single batch, got {media} and {current_media}")

        # submit the data to the pipeline which supports the media type
        indices = self._in_pools[media].submit(in_data_list)
        # collect the results
        while True:
            #TODO: timeout should be runtime configurable
            results = self._outputs[media].collect(indices, timeout=3)
            if results is None:
                logger.info("DeepstreamBackend: No more data from this batch")
                break
            out_data_list = []
            for result, pass_through_data in zip(results, pass_through_list):
                if pass_through_data:
                    result.update(pass_through_data)

                out_data = dict()
                for o in self._output_names:
                    if o in result:
                        out_data[o] = result[o]
                    else:
                        out_data[o] = None
                out_data_list.append(out_data)
            yield out_data_list
            # No consecutive inference results for image
            if media == "image":
                break

    def stop(self):
        for input in self._in_pools.values():
            input.stop()
        for pipeline in self._pipelines.values():
            pipeline.stop()
        for pipeline in self._pipelines.values():
            pipeline.join()

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