import json
from dataclasses import asdict
from config import global_config
from .model import GenericInference
from lib.utils import create_jinja2_env, convert_list
from lib.responder import ResponderBase
import re
import numpy as np
import torch
from typing import Dict, Any
from fastapi.responses import StreamingResponse
import asyncio

class Responder(ResponderBase):
    def __init__(self):
        super().__init__()
        self._inference = GenericInference()
        self._inference.initialize()
        self._lock = asyncio.Lock()

        # initialize the action map
        {% for responder in responders %}
        self._action_map["{{ responder.operation }}"] = self.{{ responder.name }}
        {% endfor %}

    def process_request(
            self,
            responder : str,
            request,
    ):
        # the helper function transforms the HTTP request to a dictionary that satisfies the inference input schema
        return super().process_request(responder, request)

    def process_response(self, responder: str, request, response: Dict[str, Any]):
        accept = request.headers.get("accept", "")
        streaming = "application/x-ndjson" in accept
        # transform numpy ndarray or tensor to universal value types for inference
        if responder == "infer":
            type_map = { i.name: i.data_type for i in global_config.output}
            for name in response:
                if not name in type_map:
                    self.logger.error(f"Unexpected output: {name}")
                    continue
                expected_type = type_map[name]
                value = response[name]
                if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                    l = value.tolist()
                    if expected_type == "TYPE_STRING" and value.dtype != np.string_:
                        response[name] = convert_list(l, lambda i: i.decode("utf-8", "ignore"))
                    elif len(response[name].shape) == 1 and len(l) == 1:
                        response[name] = l[0]
                    else:
                        response[name] = l

        # the helper function transforms the inference output to a json string
        json_string = super().process_response(responder, request, response)
        self.logger.debug(f"Sending json payload: {json_string}")

        return (json.dumps(json.loads(json_string), separators=(',', ':')) + "\n") if streaming else json.loads(json_string)

    async def take_action(self, action_name:str, **kwargs):
        action = self._action_map.get(action_name, None)
        if not action:
            raise ValueError(f"Unknown action: {action_name}")
        return await action(**kwargs)

{% for responder in responders %}
{{ responder.implementation }}
{% endfor %}
