from common.config import config
from common.inference import ModelBackend, Inference, Error
import triton_python_backend_utils as pb_utils
from omegaconf import OmegaConf
import queue
from typing import List
from common.utils import get_logger
import os

logger = get_logger(__name__)

# environment variables
CHECKPOINTS_DIR = os.getenv("CHECKPOINTS_DIR", "/workspace/checkpoints")
PIPELINE_GPU_ID=int(os.getenv("PIPELINE_GPU_ID", "0"))


class TrtLLMBackend(ModelBackend):
    def __init__(self, model_name:str, input_names: List[str], output_names: List[str]):
        self._model_name = model_name
        self._input_names = input_names
        self._output_names = output_names
        logger.debug(f"TrtLLMBackend created for {model_name} with inputs {input_names} and outputs {output_names}")

    def __call__(self, **kwargs):
        logger.debug(f"TrtLLMBackend {self._model_name} triggerred with {kwargs}")

        llm_request = pb_utils.InferenceRequest(
            model_name = self._model_name,
            requested_output_names = self._output_names,
            inputs = [pb_utils.Tensor(k, v) if not isinstance(v, pb_utils.Tensor) else v for k, v in kwargs.items() if v]
        )
        finish_reason = ""
        buffer_map = {n: [] for n in self._output_names}
        for idx, llm_response in enumerate(llm_request.exec(decoupled=True)):
            if llm_response.has_error():
                yield Error(message=f"{llm_response.error().message()}, stream_id={idx}")
            if not llm_response.output_tensors():
                continue
            for name in self._output_names:
                output = pb_utils.get_output_tensor_by_name(llm_response, name).as_numpy().flatten().tolist()
                if not output:
                    finish_reason = "stop"
                buffer_map[name] += output
            yield buffer_map
        if all(v for k, v in buffer_map.items()):
            return buffer_map



class TritonPythonModel(Inference):
    """The top level python model that drives the inference flow"""

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        # create a minimum config
        auto_complete_model_config.set_max_batch_size(1)
        auto_complete_model_config.set_dynamic_batching()
        for input in config.input:
            auto_complete_model_config.add_input(OmegaConf.to_object(input))
        for output in config.output:
            auto_complete_model_config.add_output(OmegaConf.to_object(output))
        return auto_complete_model_config

    def initialize(self, args):
        super().initialize(check_point_dir=CHECKPOINTS_DIR)
        logger.info(f"CHECKPOINTS_DIR: {CHECKPOINTS_DIR}")
        for operator in self._operators:
            model_config = next((m for m in config.inference.models if m.name == operator.model_name), None)
            if model_config.backend == "tensorrtllm":
                trtllm = TrtLLMBackend(
                    model_name=model_config.name,
                    input_names=[i.name for i in model_config.input],
                    output_names=[i.name for i in model_config.output],
                )
                self._submit(operator, trtllm)

    async def execute(self, requests):
        """ execute a list of requests"""
        logger.debug(f"received {len(requests)} requests")
        for request in requests:
            response_sender = request.get_response_sender()
            if request.is_cancelled():
                response_sender.send(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError("request is cancelled", pb_utils.TritonError.CANCELLED)),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                continue
            for input in self._inputs:
                input.put({n: pb_utils.get_input_tensor_by_name(request, n) for n in input.names})
            for output in self._outputs:
                while True:
                    try:
                        out_data = output.get()
                        if isinstance(out_data, Error):
                            error = pb_utils.TritonError(f"steam llm_response, error received: {out_data.message}")
                            response_sender.send(
                                pb_utils.InferenceResponse(error=error),
                                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                            )
                            return
                        triton_data = []
                        for name, data in zip(output.names, out_data):
                            triton_data.append(pb_utils.Tesor(name, data))
                        response_sender.send(pb_utils.InferenceResponse(triton_data))
                    except queue.Empty:
                        break
            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    def finalize(self):
        super().finalize()

