from lib.inference import ModelBackend
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner
from typing import List

class PolygraphBackend(ModelBackend):
    """Python TensorRT Backend from polygraph"""
    def __init__(self, model_name:str, output_names: List[str], engine_file: str, device_id: int=0):
        self._model_name = model_name
        self._output_names = output_names
        self._device_id = device_id
        logger.debug(f"PolygraphBackend created for {model_name} to generate {self._output_names}")
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

