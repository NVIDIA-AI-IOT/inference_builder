from pyservicemaker import Pipeline, Flow, BufferProvider, RenderMode, BatchMetadataOperator, Probe, as_tensor, ColorFormat
from typing import Dict
from queue import Queue, Empty

class TensorInput(BufferProvider):

    def __init__(self, height, width):
        super().__init__()
        self.width = width
        self.height = height
        self.format = "RGB"
        self.framerate = 1
        self.device = 'gpu'
        self._queue = Queue()

    def handle_metadata(self, batch_meta):
        result = dict()
        for frame_meta in batch_meta.frame_items:
            for user_meta in frame_meta.tensor_items:
                for n, tensor in user_meta.as_tensor_output().get_layers().items():
                    tensor = torch.utils.dlpack.from_dlpack(tensor.clone()).to('cpu')
                    result[n] = tensor
        self._queue.put(result)

    def send(self, data):
        self._queue.put(data)

class TensorOutput(BatchMetadataOperator):
    def __init__(self):
        super().__init__()
        self._queue = Queue(maxsize=1)

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            for user_meta in frame_meta.tensor_items:
                for n, tensor in user_meta.as_tensor_output().get_layers().items():
                    print(n)
                    tensor = tensor.clone()
                    print(tensor.shape)
                    print(tensor.strides)
                    print(tensor.dtype)
                    print(tensor.size)
                    torch_tensor = torch.utils.dlpack.from_dlpack(tensor).to('cpu')
                    print(torch_tensor)
                    self._queue.put(torch_tensor)

    def get(self):
        return self._queue.get()

class DeepstreamBackend(ModelBackend):
    """Deepstream backend using pyservicemaker"""
    def __init__(self, model_config:Dict, device_id: int=0):
        super().__init__(model_config)
        self._model_name = model_config["name"]
        self._output_names = [o['name'] for o in model_config['output']]
        self._device_id = device_id

        if len(model_config['input']) != 1:
            raise Exception("Deepstream pipeline handles one input")
        dims = model_config['input'][0]['dims']
        d = (dims[1], dims[2]) if dims[0] == 3 else (dims[0], dims[1])
        if "parameters" not in model_config or "infer_config_path" not in model_config["parameters"]:
            raise Exception("Deepstream pipeline requires infer_config_path")
        infer_config_path = model_config["parameters"]['infer_config_path']
        infer_element = model_config['backend'].split('/')[-1]
        with_triton = infer_element == 'nvinferserver'
        self._tensor_out = TensorOutput()
        self._tensor_input = TensorInput(d[0], d[1])
        self._pipeline = Pipeline(f"deepstream-{self._model_name}")

        # build the inference flow
        flow = Flow(self._pipeline)
        flow.inject([self._tensor_input]).batch().infer(infer_config_path, with_triton).attach(Probe('tensor_retriver', self._tensor_out)).render(RenderMode.DISCARD, enable_osd=False)
        self._pipeline.start()
        logger.debug(f"DeepstreamBackend created for {self._model_name} to generate {self._output_names}")


    def __call__(self, *args, **kwargs):
        logger.debug(f"DeepstreamBackend {self._model_name} triggerred with  {args if args else kwargs}")
        in_data_list = args if args else [kwargs]
        for item in in_data_list:
            for key in item:
                tensor = item[key]
                if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                    yield Error("Deepstream pipeline only accept GPU tensors")
            self._tensor_input.send(tensor)
            result = self._tensor_out.get()
            if not all([key in result for key in self._output_names]):
                logger.error(f"Not all the expected output in {self._output_names} are not found in the result")
                continue
            yield { o: result[o] for o in self._output_names}

    def stop(self):
        self._tensor_input.send(Stop())
        self._pipeline.wait()
