import base64
import json
from typing import Dict, Optional, List, Union
import numpy as np
from inferencemodeltoolkit.interfaces.fastapi import FastApiTritonInterface
from data_model import ChatRequest, ChatCompletion, ChatCompletionChunk, ChoiceChunk, Message, Choice, Usage
from common.config import global_config
from omegaconf.errors import ConfigKeyError
from jinja2 import Template
import re


class Interface(FastApiTritonInterface):
    def process_request(self, request: ChatRequest, headers: Dict[str, str]):
        result = json.loads(request.model_dump_json())
        # load the input config if there is any
        input_config = None
        try:
            input_config = global_config.projections.input
        except ConfigKeyError:
            pass
        if input_config is not None:
            if hasattr(input_config, 'templates'):
                tpl = base64.b64decode(input_config.templates['request']).decode()
                json_string = Template(tpl).render(result)
                result = json.loads(json_string)
                for key, value in input_config.templates.items():
                    if key in result:
                        tpl = base64.b64decode(value).decode()
                        result[key] = Template(tpl).render(data=result[key])
            if hasattr(input_config, 'filters'):
                for key, filter in input_config.filters.items():
                    if not key in result:
                        continue
                    text = result[key]
                    if not isinstance(text, str):
                        continue
                    regx = filter[0]
                    keys = filter[1:]
                    matches = re.findall(regx, text)
                    for match in matches:
                        if len(keys) != len(match):
                            continue
                        for x, y in zip(keys, match):
                            if x == '-':
                                # '-' represents replacing
                                result[key] = text.replace(y, "", 1)
                            if not x:
                                continue
                        result[x].append(y)
        return result

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