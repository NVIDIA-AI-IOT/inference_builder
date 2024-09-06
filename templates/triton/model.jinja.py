from common.config import global_config
import triton_python_backend_utils as pb_utils
from omegaconf import OmegaConf
from common.utils import get_logger
import torch
import json
import os
import numpy as np

logger = get_logger(__name__)

CHECKPOINTS_DIR = os.getenv("CHECKPOINTS_DIR", "/workspace/checkpoints")

{{ backend }}

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
        self._model_backend = None
        self._model_config = OmegaConf.to_container(model_config)
        self._in_config = {i["name"]: i for i in self._model_config["input"]}
        self._out_config = {o["name"]: o for o in self._model_config["output"]}
        self._model_repo = args["model_repository"]
        self._device_id = int(json.loads(args["model_instance_device_id"]))
        # create backend
        if "tensorrt" in model_config.backend:
            if not hasattr(model_config, "tensorrt_engine"):
                raise Exception(f"{model_name}: Engine file must be specified for tensorrt backend")
            engine_path = os.path.join(CHECKPOINTS_DIR, model_config.tensorrt_engine)
            self._model_backend = TensorRTBackend(model_name, next(iter(self._out_config)), engine_path, self._device_id)

    def execute(self, requests):
        """
        execute each request
        """
        logger.info(f"Model {self._model_config['name']} received {len(requests)} requests")

        r_list = []
        responses = []
        for request in requests:
            inputs = {k: None for k in self._in_config}
            for name, config in self._in_config.items():
                dims = config['dims']
                pb_tensor = pb_utils.get_input_tensor_by_name(request, name)
                if pb_utils.Tensor.is_cpu(pb_tensor):
                    tensor = pb_tensor.as_numpy()
                    # auto reshape
                    if len(dims) > 1 and len(tensor.shape) == (len(dims)+1):
                        tensor = np.squeeze(tensor, 0)
                else:
                    tensor = torch.utils.dlpack.from_dlpack(pb_tensor.to_dlpack())
                    # auto reshape
                    if len(dims) > 1 and len(tensor.shape) == (len(dims)+1):
                        tensor = torch.squeeze(tensor, 0)
                inputs[name] = tensor

            for output in self._model_backend(**inputs):
                r_list.append(output)
        for r in r_list:
            logger.info(f"Model {self._model_config['name']} generating responses {r}")
            for k in r:
                v = r[k]
                force_cpu = True if "force_cpu" in self._out_config[k] and self._out_config[k]["force_cpu"] else False
                if isinstance(v, torch.Tensor):
                    tensor = pb_utils.Tensor(k, v.cpu().numpy()) if force_cpu else pb_utils.Tensor.from_dlpack(k, torch.utils.dlpack.to_dlpack(v))
                elif isinstance(v, np.ndarray):
                    tensor = pb_utils.Tensor(k, v)
                elif hasattr(v, "__dlpack__") and hasattr(v, "__dlpack_device__"):
                    tensor =  pb_utils.Tensor(k, np.from_dlpack(v)) if force_cpu else pb_utils.Tensor.from_dlpack(k, v)
                else:
                    responses.append(
                        pb_utils.InferenceResponse(
                            pb_utils.TritonError(f"Unsupported tensor format from the backend. {v}")
                        )
                    )
                    break
                r[k] = tensor
            responses.append(pb_utils.InferenceResponse(output_tensors=[r[k] for k in r]))

        return responses