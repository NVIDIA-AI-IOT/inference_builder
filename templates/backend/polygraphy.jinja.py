from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

class PolygraphBackend(ModelBackend):
    """Python TensorRT Backend from polygraph"""
    def __init__(self, model_config:Dict, device_id: int=0):
        super().__init__(model_config, device_id)
        self._model_name = model_config["name"]
        self._output_names = [o['name'] for o in model_config['output']]

        logger.debug(f"PolygraphBackend created for {self._model_name} to generate {self._output_names}")
        if "tensorrt_engine" not in model_config:
            raise("PolygraphBackend requires a path to tensorrt_engine")
        engine_file = model_config["tensorrt_engine"]
        engine = EngineFromBytes(BytesFromPath(engine_file))
        self._trt_runner = TrtRunner(engine)
        self._trt_runner.activate()
        logger.info(f"TensorRT runtime created from {engine_file}")


    def __call__(self, *args, **kwargs):
        logger.debug(f"PolygraphBackend {self._model_name} triggerred with  {args if args else kwargs}")
        in_data_list = args if args else [kwargs]
        for item in in_data_list:
            for key in item:
                tensor = item[key]
                if isinstance(tensor, np.ndarray):
                    item[key] = torch.from_numpy(tensor).to(self._device_id)

            result = self._trt_runner.infer(item)
            if not all([key in result for key in self._output_names]):
                logger.error(f"Not all the expected output in {self._output_names} are not found in the result")
                continue
            yield { o: result[o] for o in self._output_names}

