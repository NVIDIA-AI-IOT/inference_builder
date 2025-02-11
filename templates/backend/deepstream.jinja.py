from pyservicemaker import Pipeline, Flow, BufferProvider, Buffer, RenderMode, BatchMetadataOperator, Probe
from typing import Dict, List
from queue import Queue, Empty
from dataclasses import dataclass, field

class TensorInput(BufferProvider):

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

class TensorInputPool:

    def __init__(self, height, width, formats, batch_size):
        self._inputs = [TensorInput(width, height, format) for _ in range(batch_size) for format in formats]

    @property
    def instances(self):
        return self._inputs

    def submit(self, data: List):
        indices = []
        for item in data:
            format = item.pop('format', None).upper()
            if format == 'JPG':
                format = 'JPEG'
            for key in item:
                tensor = item[key]
            # try find the free for the specific format
            i, tensor_input = next(((i, x) for i,x in enumerate(self._inputs) if x.format == format and x.queue.empty()), (-1, None))
            if tensor_input is None:
                i, tensor_input = next(((i, x) for i,x in enumerate(self._inputs) if x.format == format), (-1, None))
            indices.append(i)
            if tensor_input is not None:
                tensor_input.send(tensor)
            else:
                logger.error(f"Format {format} is not supported")
        return indices

class TensorOutput(BatchMetadataOperator):
    def __init__(self, n_outputs):
        super().__init__()
        self._queues = [Queue() for _ in range(n_outputs)]

    def handle_metadata(self, batch_meta):
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
    seg_map: list[int] = field(default_factory=list)

class MetadataOutput(BatchMetadataOperator):
    def __init__(self, n_outputs, output_name, d):
        super().__init__()
        self._queues = [Queue() for _ in range(n_outputs)]
        self._output_name = output_name
        self._shape = d

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            metadata = DeepstreamMetadata()
            q = self._queues[frame_meta.pad_index]
            for object_meta in frame_meta.object_items:
                left = int(object_meta.rect_params.left)
                top = int(object_meta.rect_params.top)
                width = int(object_meta.rect_params.width)
                height = int(object_meta.rect_params.height)
                metadata.shape = [self._shape[0], self._shape[1]]
                metadata.bboxes.append([left, top, left + width, top + height])
                metadata.probs.append(object_meta.confidence)
                metadata.labels.append(object_meta.label)
            for user_meta in frame_meta.tensor_items:
                seg_meta = user_meta.as_tensor_output()
                if seg_meta:
                    metadata.shape = [seg_meta.heigh, seg_meta.width]
                    metadata.seg_map = seg_meta.class_map
                    metadata.probs = seg_meta.class_probabilities_map
                # only one seg meta is expected on one frame
                break
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

        if len(self._output_names) > 1 and self._output_types[0] == "TYPE_CUSTOM_DS_METADATA":
            raise Exception(f"No more than one output is allowed for DS metadata!")
        tensor_output = False if self._output_types[0] == "TYPE_CUSTOM_DS_METADATA" else True
        dims = model_config['input'][0]['dims']
        d = (dims[1], dims[2]) if dims[0] == 3 else (dims[0], dims[1])
        if "parameters" not in model_config or "infer_config_path" not in model_config["parameters"]:
            raise Exception("Deepstream pipeline requires infer_config_path")
        infer_config_path = model_config["parameters"]['infer_config_path']
        infer_element = model_config['backend'].split('/')[-1]
        with_triton = infer_element == 'nvinferserver'
        self._formats = ['JPEG', 'PNG']
        self._in_pool = TensorInputPool(d[0], d[1], self._formats, self._max_batch_size)
        n_output = self._max_batch_size * len(self._formats)
        self._out = TensorOutput(n_output) if tensor_output else MetadataOutput(n_output, self._output_names[0], d)
        self._pipeline = Pipeline(f"deepstream-{self._model_name}")

        # build the inference flow
        flow = Flow(self._pipeline)
        probe = Probe('tensor_retriver', self._out)
        flow = flow.inject(self._in_pool.instances).decode().batch(batch_size=self._max_batch_size, batched_push_timeout=1000, live_source=False, width=d[1], height=d[0])
        flow = flow.infer(infer_config_path, with_triton, batch_size=self._max_batch_size).attach(probe).render(RenderMode.DISCARD, enable_osd=False)
        self._pipeline.start()
        logger.info(f"DeepstreamBackend created for {self._model_name} to generate {self._output_names}, output tensor: {tensor_output}")


    def __call__(self, *args, **kwargs):
        logger.debug(f"DeepstreamBackend {self._model_name} triggerred with  {args if args else kwargs}")
        in_data_list = args if args else [kwargs]
        indices = self._in_pool.submit(in_data_list)
        for result in self._out.get(indices):
            yield { o: result[o] for o in self._output_names } if result else Error("Error")

    def stop(self):
        self._tensor_input.send(Stop())
        self._pipeline.wait()
