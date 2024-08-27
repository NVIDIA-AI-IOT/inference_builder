from common.config import global_config
from common.inference import ModelBackend, Inference, Error
import triton_python_backend_utils as pb_utils
from omegaconf import OmegaConf
import queue
from typing import List
from common.utils import get_logger
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
            inputs = [pb_utils.Tensor(k, v) for k, v in kwargs.items()]
        )
        finish_reason = ""
        for idx, llm_response in enumerate(llm_request.exec(decoupled=True)):
            expected = {n: None for n in self._output_names}
            if llm_response.has_error():
                yield Error(message=f"{llm_response.error().message()}, stream_id={idx}")
            if not llm_response.output_tensors():
                continue
            for name in expected:
                output = pb_utils.get_output_tensor_by_name(llm_response, name)
                if not output:
                    finish_reason = "stop"
                expected[name] = output
            logger.debug(f"TrtLLMBackend saved inference results to: {expected}")
            yield expected
        if all(v for k, v in expected.items()):
            return expected



class TritonPythonModel(Inference):
    """The top level python model that drives the inference flow"""

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        # create a minimum config
        auto_complete_model_config.set_max_batch_size(1)
        auto_complete_model_config.set_dynamic_batching()
        auto_complete_model_config.set_model_transaction_policy({"decoupled": True})
        for input in global_config.input:
            auto_complete_model_config.add_input(OmegaConf.to_object(input))
        for output in global_config.output:
            auto_complete_model_config.add_output(OmegaConf.to_object(output))
        return auto_complete_model_config

    def initialize(self, args):
        super().initialize(check_point_dir=CHECKPOINTS_DIR)
        logger.info(f"CHECKPOINTS_DIR: {CHECKPOINTS_DIR}")
        for operator in self._operators:
            model_config = next((m for m in global_config.models if m.name == operator.model_name), None)
            if model_config.backend == "tensorrtllm":
                trtllm = TrtLLMBackend(
                    model_name=model_config.name,
                    input_names=[i.name for i in model_config.input],
                    output_names=[i.name for i in model_config.output],
                )
                self._submit(operator, trtllm)
        # thread executor for async bridge
        self._async_executor = ThreadPoolExecutor(max_workers=len(self._outputs))
        # async queues
        self._async_outputs = [asyncio.Queue() for o in self._outputs]

    async def execute(self, requests):
        """ execute a list of requests"""
        async def async_put(queue, item):
            await queue.put(item)
        def thread_to_async_bridge(thread_queue, async_queue, loop):
            item = thread_queue.get()
            if not item:
                logger.error(f"Failed to read from data flow: {item}")
                return
            asyncio.run_coroutine_threadsafe(async_put(async_queue, item), loop)

        logger.info(f"received {len(requests)} requests")
        for request in requests:
            response_sender = request.get_response_sender()
            if request.is_cancelled():
                response_sender.send(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError("request is cancelled", pb_utils.TritonError.CANCELLED)),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                continue
            for input in self._inputs:
                logger.info(f"Submitting request {request}")
                input.put({n: pb_utils.get_input_tensor_by_name(request, n) for n in input.in_names})
            # fetch result
            loop = asyncio.get_event_loop()
            for a_output, output in zip(self._async_outputs, self._outputs):
                self._async_executor.submit(thread_to_async_bridge, output, a_output, loop)
            response_data = dict()
            for a_output, output in zip(self._async_outputs, self._outputs):
                try:
                    out_data = await a_output.get()
                    logger.info(f"Got output data: {out_data}")
                    if isinstance(out_data, Error):
                        error = pb_utils.TritonError(f"steam llm_response, error received: {out_data.message}")
                        response_sender.send(
                            pb_utils.InferenceResponse(error=error),
                            flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                        )
                        return
                    response_data.update(out_data)
                except Exception as e:
                    logger.exception(e)
                response_sender.send(pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(name, tensor) for name, tensor in response_data.items()
                    ]
                ))
            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    def finalize(self):
        super().finalize()

