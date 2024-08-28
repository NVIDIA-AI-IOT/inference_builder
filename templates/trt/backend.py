from typing import List
from common.inference import ModelBackend
from common.utils import get_logger

logger = get_logger(__name__)

class TensorRTBackend(ModelBackend):
    """Python TensorRT Backend"""
    def __init__(self, model_name:str, input_names: List[str], output_names: List[str]):
        self._model_name = model_name
        self._input_names = input_names
        self._output_names = output_names
        logger.debug(f"TensorRTBackend created for {model_name} with inputs {input_names} and outputs {output_names}")

    def __call__(self, **kwargs):
        result = dict()
        logger.debug(f"TensorRTBackend {self._model_name} triggerred with {kwargs}")

        return result