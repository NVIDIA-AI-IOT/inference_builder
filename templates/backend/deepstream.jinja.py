from pyservicemaker import Pipeline, Flow, BufferProvider, Buffer, RenderMode, BatchMetadataOperator, Probe
from typing import Dict, List
from queue import Queue, Empty

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
        result = dict()
        for frame_meta in batch_meta.frame_items:
            queue = self._queues[frame_meta.pad_index]
            for user_meta in frame_meta.tensor_items:
                for n, tensor in user_meta.as_tensor_output().get_layers().items():
                    torch_tensor = torch.utils.dlpack.from_dlpack(tensor).to('cpu')
                    result[n] = torch_tensor
            queue.put(result)


    def get(self, indices: List):
        return [self._queues[i].get() if i >= 0 else None for i in indices]

class DeepstreamBackend(ModelBackend):
    """Deepstream backend using pyservicemaker"""
    def __init__(self, model_config:Dict, device_id: int=0):
        super().__init__(model_config, device_id)
        self._max_batch_size = model_config["max_batch_size"]
        self._model_name = model_config["name"]
        self._output_names = [o['name'] for o in model_config['output']]

        dims = model_config['input'][0]['dims']
        d = (dims[1], dims[2]) if dims[0] == 3 else (dims[0], dims[1])
        if "parameters" not in model_config or "infer_config_path" not in model_config["parameters"]:
            raise Exception("Deepstream pipeline requires infer_config_path")
        infer_config_path = model_config["parameters"]['infer_config_path']
        infer_element = model_config['backend'].split('/')[-1]
        with_triton = infer_element == 'nvinferserver'
        self._formats = ['JPEG', 'PNG']
        self._in_pool = TensorInputPool(d[0], d[1], self._formats, self._max_batch_size)
        self._tensor_out = TensorOutput(self._max_batch_size * len(self._formats))
        self._pipeline = Pipeline(f"deepstream-{self._model_name}")

        # build the inference flow
        flow = Flow(self._pipeline)
        probe = Probe('tensor_retriver', self._tensor_out)
        flow = flow.inject(self._in_pool.instances).decode().batch(batch_size=self._max_batch_size, batched_push_timeout=1000, live_source=False)
        flow = flow.infer(infer_config_path, with_triton, batch_size=self._max_batch_size).attach(probe).render(RenderMode.DISCARD, enable_osd=False)
        self._pipeline.start()
        logger.debug(f"DeepstreamBackend created for {self._model_name} to generate {self._output_names}")


    def __call__(self, *args, **kwargs):
        logger.debug(f"DeepstreamBackend {self._model_name} triggerred with  {args if args else kwargs}")
        in_data_list = args if args else [kwargs]
        indices = self._in_pool.submit(in_data_list)
        for result in self._tensor_out.get(indices):
            yield { o: result[o] for o in self._output_names } if result else Error("Error")

    def stop(self):
        self._tensor_input.send(Stop())
        self._pipeline.wait()
