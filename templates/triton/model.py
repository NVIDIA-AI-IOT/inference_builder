from common.config import global_config
import triton_python_backend_utils as pb_utils
from omegaconf import OmegaConf
from common.utils import get_logger

logger = get_logger(__name__)

class TritonPythonModel:
    def initialize(self, args):
        """
        This function allows
        the model to initialize any state associated with this model.
        """
        model_name = args["model_name"]
        model_config = next((m for m in global_config.models if m.name == model_name), None)
        if model_config is None:
            raise Exception("Model config not found")
        self._model_config = OmegaConf.to_container(model_config)
        self._model_repo = args["model_repository"]

    def execute(self, requests):
        """
        execute each request
        """
        logger.info(f"Model {self._model_config['name']} received {len(requests)} requests")
        for request in requests:
            inputs = {n: pb_utils.get_input_tensor_by_name(request, n) for n in input.in_names}
            logger.debug(f"Input extracted: {inputs}")

        responses = []
        return responses