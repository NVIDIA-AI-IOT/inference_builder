{#
 SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
#}
{{ license }}

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
import dataclasses

logger = get_logger(__name__)

{% for backend in backends %}
{{ backend }}
{% endfor %}

class GenericInference(InferenceBase):
    """The class that drives a generic inference flow"""

    def _create_backend(self, backend_spec: List[str], model_config: Dict, model_home: str):
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
        elif backend_spec[0] == 'vllm':
            backend_class = VLLMBackend
        else:
            return None
        return backend_class(model_config, model_home)

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

        logger.debug(f"Received request {request}")
        matched = [[n for n in input.in_names if n in request] for input in self._inputs]
        reshuffled = sorted(range(len(matched)), key=lambda x: len(matched[x]), reverse=True)
        for i in reshuffled:
            input = self._inputs[i]
            # select the tensors for the input
            tensors = { n: request[n] for n in input.in_names if n in request and request[n]}

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

        # start the async bridges to fetch the results and deposit them to the async queues
        loop = asyncio.get_event_loop()
        for a_output, output in zip(self._async_outputs, self._outputs):
            self._async_executor.submit(thread_to_async_bridge, output, a_output, loop)

        # Wait for all the results from one inference request
        while not self._stop_event.is_set():
            try:
                logger.debug("Waiting for tensors from async queue")
                response_data = dict()
                results = await asyncio.gather(*(ao.get() for ao in self._async_outputs))
                error = False
                for data in results:
                    logger.debug(f"Got output data: {data}")
                    if isinstance(data, Error):
                        logger.warning(f"Got Error: {data.message}")
                        error = True
                        break
                    elif isinstance(data, Stop):
                        logger.info(f"Got Stop: {data.reason}")
                        return
                    # collect the output
                    for k, v in data.items():
                        response_data[k] = v
                if not error:
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
                logger.warning(f"Input settings invalid for the processor: {processor.name}")
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
                    elif dataclasses.is_dataclass(v):
                        value_list.append(dataclasses.asdict(v))
                    else:
                        value_list.append(v)
                processed[key] = value_list
            else:
                if isinstance(value, np.ndarray):
                    processed[key] = value.tolist()
                elif isinstance(value, torch.Tensor):
                    processed[key] = value.tolist()
                elif dataclasses.is_dataclass(value):
                    processed[key] = dataclasses.asdict(value)
        return processed