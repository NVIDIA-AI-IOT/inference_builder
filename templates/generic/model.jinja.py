from config import global_config
from lib.inference import *
from lib.utils import *
from omegaconf import OmegaConf
import json
import os
from typing import List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from pathlib import Path

logger = get_logger(__name__)

{% for backend in backends %}
{{ backend }}
{% endfor %}

class GenericInference(InferenceBase):
    """The class that drives a generic inference flow"""


    def initialize(self, *args):
        model_repo = global_config.model_repo
        logger.info(f"Model Repository: {model_repo}")
        super().initialize(model_repo)
        for operator in self._operators:
            model_config = next((m for m in global_config.models if m.name == operator.model_name), None)
            backend_spec = model_config.backend.split('/')
            backend_instance = None
            backend_class = None
            if backend_spec[0] == 'triton':
                backend_class = TritonBackend
            elif backend_spec[0] == 'deepstream':
                backend_class = DeepstreamBackend
            elif backend_spec[0] == 'polygraphy':
                backend_class = PolygraphBackend
            elif backend_spec[0] == 'tensorrtllm':
                backend_class = TensorRTLLMBackend
            elif backend_spec[0] == 'dummy':
                backend_class = DummyBackend
            elif backend_spec[0] == 'pytorch':
                backend_class = PytorchBackend
            else:
                raise Exception(f"Backend {model_config.backend} not supported")
            backend_instance = backend_class(
                model_config=OmegaConf.to_container(model_config),
                model_home=os.path.join(model_repo, operator.model_name)
            )
            self._submit(operator, backend_instance)
        # post processing:
        self._processors = []
        if hasattr(global_config, "postprocessors"):
            configs = OmegaConf.to_container(global_config.postprocessors)
            for config in configs:
                if config["kind"] == "custom":
                    self._processors.append(
                        CustomProcessor(config, model_repo)
                    )

        # thread executor for async bridge
        self._async_executor = ThreadPoolExecutor(max_workers=len(self._outputs))
        # async queues
        self._async_outputs = [asyncio.Queue() for o in self._outputs]
        self._stop_event = threading.Event()
        logger.info(f"GenericInference {global_config.name} initialized:")
        logger.info(f"Inputs: {[f.o_names for f in self._inputs]}, Outputs:  {[f.o_names for f in self._outputs]}")

    async def execute(self, request):
        """ execute a list of requests"""
        async def async_put(queue, item):
            await queue.put(item)
        def thread_to_async_bridge(thread_queue, async_queue, loop):
            while not self._stop_event.is_set():
                try:
                    item = thread_queue.get()
                    asyncio.run_coroutine_threadsafe(async_put(async_queue, item), loop)
                except Empty:
                    continue
            logger.info(f"thread_to_async_bridge {thread_queue} stopped")

        logger.info(f"Received request {request}")
        matched = [[n for n in input.in_names if n in request] for input in self._inputs]
        reshuffled = sorted(range(len(matched)), key=lambda x: len(matched[x]), reverse=True)
        for i in reshuffled:
            input = self._inputs[i]
            # select the tensors for the input
            tensors = { n: request[n] for n in input.in_names if n in request and request[n] is not None }

            # the tensors need to be transformed to generic type
            for name in tensors:
                tensor = tensors[name]
                config = next((c for c in self._input_config if c['name'] == name), None)
                if config is None:
                    logger.warning(f"Invalid input parsed: {name}")
                    continue
                tensors[name] = np.array(tensor)
            if tensors:
                logger.debug(f"Injecting tensors {tensors}")
                input.put(tensors)
                input.put(Stop(reason="end"))
        # fetch result
        loop = asyncio.get_event_loop()
        for a_output, output in zip(self._async_outputs, self._outputs):
            self._async_executor.submit(thread_to_async_bridge, output, a_output, loop)
        while not self._stop_event.is_set():
            try:
                logger.debug("Waiting for tensors from async queue")
                response_data = dict()
                results = await asyncio.gather(*(ao.get() for ao in self._async_outputs))
                for data in results:
                    logger.debug(f"Got output data: {data}")
                    if isinstance(data, Error) or isinstance(data, Stop):
                        logger.info(f"Inference batch ended with {data}")
                        return
                    # collect the output
                    for k, v in data.items():
                        response_data[k] = v
                # post-process the data from all the outputs
                response_data = self._post_process(response_data)
                yield response_data
            except Exception as e:
                logger.exception(e)


    def finalize(self):
        self._stop_event.set()
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
                # correct data type
                output_config = next((c for c in self._output_config if c['name'] == key), None)
                if output_config is None:
                    logger.warning(f"Invalid output parsed: {key}")
                    continue
                data_type = output_config["data_type"]
                if isinstance(value, np.ndarray):
                    data_type = np_datatype_mapping[data_type]
                    if value.dtype != data_type:
                        processed[key]  = value.astype(data_type)
                elif isinstance(value, torch.Tensor):
                    data_type = torch_datatype_mapping[data_type]
                    if value.dtype != data_type:
                        processed[key]  = value.to(data_type)
                else:
                    processed[key] = value
        # convert numpy and torch tensors to list for server to process
        for key, value in processed.items():
            if isinstance(value, list):
                # this is a batch of data
                value_list = []
                for v in value:
                    if isinstance(v, np.ndarray):
                        value_list.append(v.tolist())
                    elif isinstance(v, torch.Tensor):
                        value_list.append(v.tolist())
                    else:
                        value_list.append(v)
                processed[key] = value_list
            else:
                if isinstance(value, np.ndarray):
                    processed[key] = value.tolist()
                elif isinstance(value, torch.Tensor):
                    processed[key] = value.tolist()
        return processed