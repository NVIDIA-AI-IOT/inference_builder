from common.config import global_config
from lib.inference import ModelBackend, CustomProcessor, InferenceBase, Error, Stop
from lib.utils import get_logger, stack_tensors_in_dict
import triton_python_backend_utils as pb_utils
from omegaconf import OmegaConf
from typing import List, Dict
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from pathlib import Path

logger = get_logger(__name__)

# environment variables
CHECKPOINTS_DIR = os.getenv("CHECKPOINTS_DIR", Path(__file__).resolve().parent.parent.parent)
PIPELINE_GPU_ID=int(os.getenv("PIPELINE_GPU_ID", "0"))


class TritonBackend(ModelBackend):
    def __init__(self, model_name:str, input_names: List[str], output_names: List[str], model_config: Dict):
        self._model_name = model_name
        self._input_names = input_names
        self._output_names = output_names
        self._model_config = model_config
        logger.debug(f"TritonBackend created for {model_name} with inputs {input_names} and outputs {output_names}")
        logger.debug(f"model_config: {model_config}")

    def __call__(self, *args, **kwargs):
        logger.debug(f"TritonBackend {self._model_name} triggerred with {args if args else kwargs}")
        in_data_list = args if args else [kwargs]
        input_config = self._model_config["input"]
        # to determine if we need to stack the input, TODO: below logic needs be more generic
        need_stack = False
        for in_data in in_data_list:
            for key, value in in_data.items():
                input_config = next((i for i in self._model_config["input"] if i['name'] == key), None)
                if input_config is None:
                    logger.error(f"Unexpected input: {key}")
                    continue
                expected_dims = len(input_config["dims"])
                if expected_dims == len(value.shape) + 1:
                    need_stack = True
                    break
        if need_stack:
            in_data_list = [stack_tensors_in_dict(in_data_list)]
        for in_data in in_data_list:
            tensors = []
            for k in in_data:
                tensor = in_data[k]
                batched = "max_batch_size" in self._model_config and self._model_config["max_batch_size"] > 0
                if isinstance(tensor, np.ndarray):
                    tensor = pb_utils.Tensor(k, np.expand_dims(tensor, 0)) if batched else pb_utils.Tensor(k, tensor)
                elif isinstance(tensor, torch.Tensor):
                    if batched:
                        tensor = tensor.unsqueeze(0)
                    tensor = pb_utils.Tensor.from_dlpack(k, torch.utils.dlpack.to_dlpack(tensor))
                elif hasattr(tensor, "__dlpack__") and hasattr(tensor, "__dlpack_device__"):
                    tensor = pb_utils.Tensor.from_dlpack(k, tensor)
                else:
                    yield Error(message="Unsupported input tensor format")
                    return
                tensors.append(tensor)

            llm_request = pb_utils.InferenceRequest(
                model_name = self._model_name,
                requested_output_names = self._output_names,
                inputs = tensors
            )
            finish_reason = "done"
            for idx, response in enumerate(llm_request.exec(decoupled=True)):
                expected = {n: None for n in self._output_names}
                if response.has_error():
                    yield Error(message=f"{response.error().message()}, stream_id={idx}")
                if not response.output_tensors():
                    continue
                for name in expected:
                    config = next((c for c in self._model_config['output'] if c['name'] == name), None)
                    dims = config['dims']
                    output = pb_utils.get_output_tensor_by_name(response, name)
                    if not output:
                        continue
                    if pb_utils.Tensor.is_cpu(output):
                        tensor = output.as_numpy()
                        if tensor.size != 0:
                            expected[name] = np.squeeze(tensor, 0) if len(tensor.shape) == (len(dims)+1) else tensor
                    else:
                        tensor = torch.utils.dlpack.from_dlpack(output.to_dlpack())
                        if tensor.numel() != 0:
                            expected[name] = torch.squeeze(tensor, 0) if len(tensor.shape) == (len(dims)+1) else tensor
                logger.debug(f"TritonBackend saved inference results to: {expected}")
                if all([expected[k] is not None for k in expected]):
                    yield expected
        yield Stop(finish_reason)
        logger.info(f"Infernece with TritonBackend {self._model_name} accomplished")

class TritonPythonModel(InferenceBase):
    """The top level python model that drives the inference flow"""

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        # create a minimum config
        auto_complete_model_config.set_max_batch_size(1)
        auto_complete_model_config.set_dynamic_batching()
        auto_complete_model_config.set_model_transaction_policy({"decoupled": True})
        for input in global_config.input:
            input_config = OmegaConf.to_container(input)
            if input_config["data_type"] == "TYPE_CUSTOM_IMAGE_BASE64":
                input_config["data_type"] = "TYPE_STRING"
            auto_complete_model_config.add_input(input_config)
        for output in global_config.output:
            output_config = OmegaConf.to_container(output)
            auto_complete_model_config.add_output(output_config)
        logger.info(f"Model configuration completed as {auto_complete_model_config.as_dict()}")
        return auto_complete_model_config

    def initialize(self, args):
        logger.info(f"CHECKPOINTS_DIR: {CHECKPOINTS_DIR}")
        super().initialize(check_point_dir=CHECKPOINTS_DIR)
        for operator in self._operators:
            model_config = next((m for m in global_config.models if m.name == operator.model_name), None)
            triton_backend = TritonBackend(
                model_name=model_config.name,
                input_names=[i.name for i in model_config.input],
                output_names=[i.name for i in model_config.output],
                model_config=OmegaConf.to_container(model_config)
            )
            self._submit(operator, triton_backend)
        # post processing:
        self._processors = []
        if hasattr(global_config, "post_processors"):
            configs = OmegaConf.to_container(global_config.post_processors)
            for config in configs:
                if config["kind"] == "custom":
                    self._processors.append(
                        CustomProcessor(config, CHECKPOINTS_DIR, global_config.name)
                    )

        # thread executor for async bridge
        self._async_executor = ThreadPoolExecutor(max_workers=len(self._outputs))
        # async queues
        self._async_outputs = [asyncio.Queue() for o in self._outputs]
        logger.info(f"Model {global_config.name} initialized:")
        logger.info(f"Inputs: {[f.o_names for f in self._inputs]}, Outputs:  {[f.o_names for f in self._outputs]}")

    async def execute(self, requests):
        """ execute a list of requests"""
        async def async_put(queue, item):
            await queue.put(item)
        def thread_to_async_bridge(thread_queue, async_queue, loop):
            while True:
                item = thread_queue.get()
                asyncio.run_coroutine_threadsafe(async_put(async_queue, item), loop)
                if not item:
                    break


        logger.info(f"Received {len(requests)} request(s)")
        for request in requests:
            response_sender = request.get_response_sender()
            if request.is_cancelled():
                response_sender.send(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError("request is cancelled", pb_utils.TritonError.CANCELLED)),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                continue
            for input in self._inputs:
                # select the tensors for the input
                tensors = {n: pb_utils.get_input_tensor_by_name(request, n) for n in input.in_names}
                # the tensors need to be transformed to generic type
                for name in tensors:
                    tensor = tensors[name]
                    config = next((c for c in self._input_config if c['name'] == name), None)
                    if config is None:
                        logger.warning(f"Invalid input parsed: {name}")
                        continue
                    dims = config['dims']
                    if pb_utils.Tensor.is_cpu(tensor):
                        tensor = tensor.as_numpy()
                        # auto reshape
                        if len(tensor.shape) == (len(dims)+1):
                            tensor = np.squeeze(tensor, 0)
                    else:
                        tensor = torch.utils.dlpack.from_dlpack(tensor.to_dlpack())
                        # auto reshape
                        if len(tensor.shape) == (len(dims)+1):
                            tensor = torch.squeeze(tensor, 0)
                    tensors[name] = tensor
                logger.debug(f"Injecting tensors {tensors}")
                input.put(tensors)
                input.put(Stop(reason="end"))
            # fetch result
            loop = asyncio.get_event_loop()
            for a_output, output in zip(self._async_outputs, self._outputs):
                self._async_executor.submit(thread_to_async_bridge, output, a_output, loop)
            stop = False
            while not stop:
                try:
                    logger.debug("Waiting for tensors from async queue")
                    response_data = dict()
                    done, _ = await asyncio.wait([ao.get() for ao in self._async_outputs], return_when=asyncio.ALL_COMPLETED)
                    for f in done:
                        data = f.result()
                        logger.debug(f"Got output data: {data}")
                        if isinstance(data, Error):
                            error = pb_utils.TritonError(f"steam llm_response, error received: {data.message}")
                            response_sender.send(
                                pb_utils.InferenceResponse(error=error),
                                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                            )
                            return
                        elif isinstance(data, Stop):
                            stop = True
                            continue
                        # collect the output
                        for k, v in data.items():
                            response_data[k] = v
                    response_data = self._post_process(response_data)
                    # response with partial data
                    response_sender.send(pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor(name, tensor) for name, tensor in response_data.items()
                        ]
                    ))
                except Exception as e:
                    logger.exception(e)

            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            logger.debug(f"Finalizing the response for requests: {request}")
        logger.debug(f"All request done")

    def finalize(self):
        super().finalize()

    def _post_process(self, data: Dict):
        processed = {k: v for k, v in data.items()}
        for processor in self._processors:
            if not all([i in data for i in processor.input]):
                logger.warning(f"Input settings invalid for the processor: {processor}")
                continue
            input = [processed.pop(i) for i in processor.input]
            logger.debug(f"Post-processor invoked with given input {input}")
            output = processor(*input)
            logger.debug(f"Post-processor generated output {output}")
            if len(output) != len(processor.output):
                logger.warning(f"Number of postprocessing output doesn't match the configuration, expecting {len(processor.output)}, while getting {len(output)}")
                continue
            # update as processed
            for key, value in zip(processor.output, output):
                processed[key] = value
        return processed