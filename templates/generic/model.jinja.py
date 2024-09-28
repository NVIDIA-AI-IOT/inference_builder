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

CHECKPOINTS_DIR = os.getenv("CHECKPOINTS_DIR", Path(__file__).resolve().parent.parent.parent)


{% for backend in backends %}
{{ backend }}
{% endfor %}

class GenericInference(InferenceBase):
    """The class that drives a generic inference flow"""


    def initialize(self, *args):
        logger.info(f"CHECKPOINTS_DIR: {CHECKPOINTS_DIR}")
        super().initialize(check_point_dir=CHECKPOINTS_DIR)
        for operator in self._operators:
            model_config = next((m for m in global_config.models if m.name == operator.model_name), None)
            backend_spec = model_config.backend.split('/')
            backend_instance = None
            backend_class = None
            if backend_spec[0] == 'triton':
                backend_class = TritonBackend
            elif backend_spec[0] == 'deepstream':
                backend_class = DeepstreamBackend
            else:
                raise Exception(f"Backend {model_config.backend} not supported")
            backend_instance = backend_class(model_config=OmegaConf.to_container(model_config))
            self._submit(operator, backend_instance)
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
        logger.info(f"GenericInference {global_config.name} initialized:")
        logger.info(f"Inputs: {[f.o_names for f in self._inputs]}, Outputs:  {[f.o_names for f in self._outputs]}")

    async def execute(self, request):
        """ execute a list of requests"""
        async def async_put(queue, item):
            await queue.put(item)
        def thread_to_async_bridge(thread_queue, async_queue, loop):
            while True:
                item = thread_queue.get()
                asyncio.run_coroutine_threadsafe(async_put(async_queue, item), loop)
                if not item:
                    break

        logger.info(f"Received request {request}")
        responses = []
        for input in self._inputs:
            # select the tensors for the input
            tensors = {n: request[n] for n in input.in_names}
            # the tensors need to be transformed to generic type
            for name in tensors:
                tensor = tensors[name]
                config = next((c for c in self._input_config if c['name'] == name), None)
                if config is None:
                    logger.warning(f"Invalid input parsed: {name}")
                    continue
                dims = config['dims']
                if len(dims) == 1 and dims[0] == 1:
                    tensor = np.array([tensor])
                else:
                    tensor = np.array(tensor)
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
                        yield { "Error": data.message }
                    elif isinstance(data, Stop):
                        stop = True
                        continue
                    # collect the output
                    for k, v in data.items():
                        response_data[k] = v
                response_data = self._post_process(response_data)
                # response with partial data
                yield response_data
            except Exception as e:
                logger.exception(e)


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