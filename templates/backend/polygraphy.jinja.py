from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

class PolygraphBackend(ModelBackend):
    """Python TensorRT Backend from polygraph"""
    def __init__(self, model_config:Dict, model_home: str, device_id: int=0):
        super().__init__(model_config, model_home, device_id)
        self._model_name = model_config["name"]
        self._output_names = [o['name'] for o in model_config['output']]

        logger.debug(f"PolygraphBackend created for {self._model_name} to generate {self._output_names}")
        if "parameters" not in model_config or "tensorrt_engine" not in model_config["parameters"]:
            raise("PolygraphBackend requires a path to tensorrt_engine")
        engine_file = model_config["parameters"]["tensorrt_engine"]
        if not os.path.isabs(engine_file):
            engine_file = os.path.join(self._model_home, engine_file)
        engine = EngineFromBytes(BytesFromPath(engine_file))
        self._trt_runner = TrtRunner(engine)
        self._trt_runner.activate()
        logger.info(f"TensorRT runtime created from {engine_file}")


    def __call__(self, *args, **kwargs):
        logger.debug(f"PolygraphBackend {self._model_name} triggerred with  {args if args else kwargs}")
        in_data = stack_tensors_in_dict(args) if args else dict(kwargs)
        for k, v in in_data.items():
            if isinstance(v, np.ndarray):
                in_data[k] = torch.from_numpy(v).to(self._device_id)
        result = self._trt_runner.infer(in_data)
        if not all([key in result for key in self._output_names]):
            logger.error(f"Not all the expected output in {self._output_names} are not found in the result")
            return
        o_data = { o: result[o] for o in self._output_names}
        logger.debug(f"PolygraphBackend {self._model_name} generated output: {o_data}")
        yield o_data

