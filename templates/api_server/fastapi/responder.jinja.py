import base64
import json
from config import global_config
from .model import GenericInference
from lib.utils import create_jinja2_env, stack_tensors_in_dict, convert_list, get_logger
from omegaconf.errors import ConfigKeyError
import re
from omegaconf import OmegaConf
import numpy as np
import torch
from typing import Dict

jinja2_env = create_jinja2_env()

class Responder:
    def __init__(self):
        self._action_map = dict()
        self._inference = GenericInference()
        self._inference.initialize()
        self._request_templates = dict()
        self._response_templates = dict()
        self.logger = get_logger(__name__)
        # load the request and response templates
        try:
            input_config = OmegaConf.to_container(global_config.server.responders)
        except ConfigKeyError:
            raise ValueError("No responders found in the config")
        for rp, value in input_config.items():
            req_tpls = value.get("requests", None)
            if req_tpls:
                self._request_templates[rp] = { k: base64.b64decode(tpl).decode() for k, tpl in req_tpls.items() if tpl}
            res_tpls = value.get("responses", None)
            if res_tpls:
                self._response_templates[rp] = { k: base64.b64decode(tpl).decode() for k, tpl in res_tpls.items() if tpl}

        # initialize the action map
        {% for responder in responders %}
        self._action_map["{{ responder.operation }}"] = self.{{ responder.name }}
        {% endfor %}

    def process_request(
            self,
            responder : str,
            request,
    ):
        result = json.loads(request.model_dump_json())

        # find the request template for the endpoint
        templates = self._request_templates.get(responder, None)
        if not templates:
            # no template to be applied on the request
            return result

        request_class = next(iter(templates.keys()))
        template = templates[request_class]
        json_string = jinja2_env.from_string(template).render(request=result)
        result = json.loads(json_string)
        print(f"request: {result}")
        # template filters on each field
        for key, value in templates.items():
            if not key in result:
                continue
            if isinstance(value, list):
                # this is regex filter for fields
                text = result[key]
                if not isinstance(text, str):
                    continue
                regx = value[0]
                keys = value[1:]
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

    def process_response(self, responder: str, request, response: Dict[str, np.ndarray]):
        self.logger.debug(f"Processing response {response}")
        type_map = { i.name: i.data_type for i in global_config.output}

        # transform numpy ndarray to universal value types
        for name in response:
            if not name in response:
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

        # Load the response template for the endpoint
        templates = self._response_templates.get(responder, None)
        if not templates:
            # no template to be applied on the response
            return response

        response_class = next(iter(templates.keys()))
        template = templates[response_class]
        json_string = jinja2_env.from_string(template).render(request=request, response=response)
        self.logger.debug(f"Sending json payload: {json_string}")

        return json.loads(json_string)

    async def take_action(self, action_name:str, *args):
        print(f"args: {args}")
        action = self._action_map.get(action_name, None)
        if not action:
            raise ValueError(f"Unknown action: {action_name}")
        return await action(*args)

{% for responder in responders %}
{{ responder.implementation }}
{% endfor %}
