from typing import Dict, Optional, List, Union
import numpy as np
from inferencemodeltoolkit.interfaces.fastapi import FastApiTritonInterface
from data_model import ChatRequest, ChatCompletion, ChatCompletionChunk, ChoiceChunk, Message, Choice, Usage
from common.config import global_config
from common.mapping import InputMapping
from omegaconf import OmegaConf


class Interface(FastApiTritonInterface):
    def process_request(self, request: ChatRequest, headers: Dict[str, str]):
        inputs = OmegaConf.to_container(global_config.input)
        i_map = dict()
        # load the input mapping if there is any
        if hasattr(global_config, "input_map"):
            i_map = OmegaConf.to_container(global_config.input_map)
        return InputMapping(inputs, request.model_dump_json(), i_map)()

    def process_response(
        self,
        request: ChatRequest,
        response: Dict[str, np.ndarray],
        previous_responses: Optional[List[Dict[str, np.ndarray]]],
        headers
    ) -> Union[ChatCompletion, ChatCompletionChunk]:
        text = response['text'][0].decode('utf-8', 'ignore')
        if request.stream:
            if previous_responses:
                prev_text = previous_responses[-1]["text"][0].decode("utf-8", "ignore")
            choice = ChoiceChunk(
                index=0,
                delta=Message(role="assistant", content=text),
                finish_reason="stop"
            )
            return ChatCompletionChunk(id=request._id, choices=[choice])

        choice = Choice(
            index=0,
            message=Message(role="assistant", content=text),
            finish_reason="stop",
        )
        usage = Usage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
        return ChatCompletion(id=request._id, choices=[choice], usage=usage)

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