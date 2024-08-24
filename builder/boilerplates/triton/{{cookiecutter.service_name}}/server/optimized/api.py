from typing import Dict, Optional, List, Union
import numpy as np
from inferencemodeltoolkit.interfaces.fastapi import FastApiTritonInterface
from data_model import ChatRequest, ChatCompletion, ChatCompletionChunk
from common.config import config
from common.mapping import InputMapping
from omegaconf import OmegaConf


class Interface(FastApiTritonInterface):
    def process_request(self, request: ChatRequest, headers: Dict[str, str]):
        inputs = OmegaConf.to_container(config.input)
        i_map = OmegaConf.to_container(config.input_map)
        return InputMapping(inputs, request.model_dump(), i_map)()

    def process_response(
        self,
        request: ChatRequest,
        response: Dict[str, np.ndarray],
        previous_responses: Optional[List[Dict[str, np.ndarray]]],
        headers
    ) -> Union[ChatCompletion, ChatCompletionChunk]:
        pass

if __name__ == "__main__":
    interface = Interface(
        triton_url="grpc://localhost:8001",
        model_name="{{cookiecutter.service_name}}",
        stream_triton=True,
        stream_http=lambda request: request.stream,
        infer_endpoint="{{cookiecutter.endpoints.infer}}",
        health_endpoint="{{cookiecutter.endpoints.health}}",
        triton_timeout_s = 60,
    )
    interface.serve()