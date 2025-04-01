from pyservicemaker import Pipeline, Flow, BufferProvider, Buffer, RenderMode, BatchMetadataOperator, Probe, as_tensor
from typing import Dict, List
from queue import Queue, Empty
from dataclasses import dataclass, field
import base64

warmup_data_0 = {
        "images": np.frombuffer(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGK6HcwNCAAA//8DTgE8HuxwEQAAAABJRU5ErkJggg=="), dtype=np.uint8),
        "mime": "image/png"
}

warmup_data_1 = {
        "images": np.frombuffer(base64.b64decode("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAgACADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD+f+iiigAooooAKKKKACiiigD/2Q=="), dtype=np.uint8),
        "mime": "image/jpeg"
}


class ImageTensorInput(BufferProvider):

    def __init__(self, height, width, format, tensor_name):
        super().__init__()
        self.width = width
        self.height = height
        self.format = format
        self.framerate = 1
        self.device = 'cpu'
        self.queue = Queue(maxsize=1)
        self._tensor_name = tensor_name

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

    @property
    def tensor_name(self):
        return self._tensor_name

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

class TensorInputPool:

    def __init__(self, height, width, formats, batch_size, image_tensor_name, mime_tensor_name, device_id, require_extra_input=False):
        self._image_inputs = [ImageTensorInput(width, height, format, image_tensor_name) for format in formats for _ in range(batch_size)]
        self._mime_tensor_name = mime_tensor_name
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
                # try find the free for the specific format
                i, image_tensor_input = next(((i, x) for i,x in enumerate(self._image_inputs) if x.format == format and x.queue.empty()), (-1, None))
                if image_tensor_input is None:
                    i, image_tensor_input = next(((i, x) for i,x in enumerate(self._image_inputs) if x.format == format), (-1, None))
                indices.append(i)
                if image_tensor_input is not None:
                    image_tensor = item.pop(image_tensor_input.tensor_name, None)
                    image_tensor_input.send(image_tensor)
                else:
                    logger.error(f"Unable to find tensor input for format {format}")
            else:
                logger.error(f"Unsupported MIME type {mime_type}")
                continue
        if self._generic_input:
            self._generic_input.send(stack_tensors_in_dict(data))
        # batched indices for each input
        return indices

class BaseTensorOutput(BatchMetadataOperator):
    def __init__(self, n_outputs):
        super().__init__()
        self._queues = [Queue() for _ in range(n_outputs)]

    def handle_metadata(self, batch_meta):
        pass

    def get(self, indices: List):
        pass

class TensorOutput(BaseTensorOutput):
    def __init__(self, n_outputs, preprocess_config_path):
        super().__init__(n_outputs)
        self._preprocess_config_path = preprocess_config_path

    def handle_metadata(self, batch_meta):
        if self._preprocess_config_path:
            for meta in batch_meta.preprocess_batch_items:
                preprocess_batch = meta.as_preprocess_batch()
                if not preprocess_batch:
                    continue
                for roi in preprocess_batch.rois:
                    result = dict()
                    q = self._queues[roi.frame_meta.pad_index]
                    for user_meta in roi.tensor_items:
                        tensor_output = user_meta.as_tensor_output()
                        if tensor_output :
                            for n, tensor in tensor_output.get_layers().items():
                                torch_tensor = torch.utils.dlpack.from_dlpack(tensor).to('cpu')
                                result[n] = torch_tensor
                    q.put(result)
        else:
            for frame_meta in batch_meta.frame_items:
                result = dict()
                q = self._queues[frame_meta.pad_index]
                for user_meta in frame_meta.tensor_items:
                    tensor_output = user_meta.as_tensor_output()
                    if tensor_output :
                        for n, tensor in tensor_output.get_layers().items():
                            torch_tensor = torch.utils.dlpack.from_dlpack(tensor).to('cpu')
                            result[n] = torch_tensor
                q.put(result)

    def get(self, indices: List):
        return [self._queues[i].get() if i >= 0 else None for i in indices]

@dataclass
class DeepstreamMetadata:
    shape: list[int] = field(default_factory=lambda: [0, 0])
    bboxes: list[list[int]] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    seg_maps: list[list[int]] = field(default_factory=list)

class PreprocessMetadataOutput(BaseTensorOutput):
    def __init__(self, n_outputs, output_name, d):
        super().__init__(n_outputs)
        self._output_name = output_name
        self._shape = d

    def handle_metadata(self, batch_meta):
        for meta in batch_meta.preprocess_batch_items:
            preprocess_batch = meta.as_preprocess_batch()
            if not preprocess_batch:
                continue
            for roi in preprocess_batch.rois:
                metadata = DeepstreamMetadata()
                q = self._queues[roi.frame_meta.pad_index]
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
                q.put({"data": metadata})

    def get(self, indices: List):
        return [{self._output_name: self._queues[i].get()} if i >= 0 else None for i in indices]

class MetadataOutput(BaseTensorOutput):
    def __init__(self, n_outputs, output_name, d):
        super().__init__(n_outputs)
        self._output_name = output_name
        self._shape = d

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            metadata = DeepstreamMetadata()
            q = self._queues[frame_meta.pad_index]
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
            q.put({"data": metadata})

    def get(self, indices: List):
        return [{self._output_name: self._queues[i].get()} if i >= 0 else None for i in indices]

class DeepstreamBackend(ModelBackend):
    """Deepstream backend using pyservicemaker"""
    def __init__(self, model_config:Dict, device_id: int=0):
        super().__init__(model_config, device_id)
        self._max_batch_size = model_config["max_batch_size"]
        self._model_name = model_config["name"]
        self._output_names = [o['name'] for o in model_config['output']]
        self._output_types = [o['data_type'] for o in model_config['output']]
        self._pass_through_tensors = []

        if len(self._output_names) > 1 and self._output_types[0] == "TYPE_CUSTOM_DS_METADATA":
            raise Exception(f"No more than one output is allowed for DS metadata!")
        tensor_output = False if self._output_types[0] == "TYPE_CUSTOM_DS_METADATA" else True
        dims = model_config['input'][0]['dims']
        d = (dims[1], dims[2]) if dims[0] == 3 else (dims[0], dims[1])
        if "parameters" not in model_config or "infer_config_path" not in model_config["parameters"]:
            raise Exception("Deepstream pipeline requires infer_config_path")
        infer_config_path = model_config["parameters"]['infer_config_path']
        preprocess_config_path = model_config["parameters"]['preprocess_config_path'] if "preprocess_config_path" in model_config["parameters"] else []
        infer_element = model_config['backend'].split('/')[-1]
        with_triton = infer_element == 'nvinferserver'
        image_tensor_name = None
        require_extra_input = False
        mime_tensor_name = None
        for input in model_config['input']:
            if input['data_type'] == 'TYPE_CUSTOM_DS_IMAGE':
                image_tensor_name = input['name']
            elif input['data_type'] == 'TYPE_CUSTOM_DS_PASSTHROUGH':
                self._pass_through_tensors.append(input['name'])
            elif input['data_type'] == 'TYPE_CUSTOM_DS_MIME':
                mime_tensor_name = input['name']
            elif not ('optional' in input and input['optional']):
                tensor_name = input['name']
                require_extra_input = True
                np_type = np_datatype_mapping[input['data_type']]
                warmup_data_0[tensor_name] = np.random.rand(*input['dims']).astype(np_type)
                warmup_data_1[tensor_name] = np.random.rand(*input['dims']).astype(np_type)
        if image_tensor_name is None:
            raise Exception("Deepstream pipeline requires at least one TYPE_CUSTOM_DS_IMAGE input")
        if mime_tensor_name is None:
            raise Exception("Deepstream pipeline requires at least one TYPE_CUSTOM_DS_MIME input")

        self._formats = ['JPEG', 'PNG']
        self._in_pool = TensorInputPool(d[0], d[1], self._formats, self._max_batch_size, image_tensor_name, mime_tensor_name, device_id, require_extra_input)
        n_output = self._max_batch_size * len(self._formats)
        if tensor_output:
            self._out = TensorOutput(n_output, preprocess_config_path)
        elif preprocess_config_path:
            self._out = PreprocessMetadataOutput(n_output, self._output_names[0], d)
        else:
            self._out = MetadataOutput(n_output, self._output_names[0], d)
        self._pipeline = Pipeline(f"deepstream-{self._model_name}")

        # build the inference flow
        flow = Flow(self._pipeline)
        probe = Probe('tensor_retriver', self._out)
        flow = flow.inject(self._in_pool.image_inputs).decode().batch(batch_size=self._max_batch_size, batched_push_timeout=1000, live_source=False, width=d[1], height=d[0])
        for config in preprocess_config_path:
            flow = flow.preprocess(config, None if not require_extra_input else lambda: self._in_pool.generic_input.generate())
        for config in infer_config_path:
            flow = flow.infer(config, with_triton, batch_size=self._max_batch_size)
        flow = flow.attach(probe).render(RenderMode.DISCARD, enable_osd=False)
        self._pipeline.start()
        logger.info(f"DeepstreamBackend created for {self._model_name} to generate {self._output_names}, output tensor: {tensor_output}")

        # warm up
        indices = self._in_pool.submit([warmup_data_0.copy() for _ in range(self._max_batch_size)])
        for result in self._out.get(indices):
            logger.info(f"Warm up 0: {result}")
        indices = self._in_pool.submit([warmup_data_1.copy() for _ in range(self._max_batch_size)])
        for result in self._out.get(indices):
            logger.info(f"Warm up 1: {result}")


    def __call__(self, *args, **kwargs):
        logger.debug(f"DeepstreamBackend {self._model_name} triggerred with  {args if args else kwargs}")
        in_data_list = args if args else [kwargs]
        pass_through_list = [dict() for _ in range(len(in_data_list))]
        for data, pass_through_data in zip(in_data_list, pass_through_list):
            for tensor_name in self._pass_through_tensors:
                pass_through_data[tensor_name] = data.pop(tensor_name, None)
        indices = self._in_pool.submit(in_data_list)
        for result, pass_through_data in zip(self._out.get(indices), pass_through_list):
            if pass_through_data:
                result.update(pass_through_data)
            # check the result and yield the output
            if result:
                out_data = dict()
                for o in self._output_names:
                    if o in result:
                        out_data[o] = result[o]
                    else:
                        out_data[o] = None
                yield out_data
            else:
                yield Error("Error")

    def stop(self):
        self._tensor_input.send(Stop())
        self._pipeline.wait()
