from nimlib.nim_inference_api_builder.http_api import HttpNIMApiInterface
from fastapi import Request
import base64
import json
import data_model
from config import global_config
from model import GenericInference
from omegaconf.errors import ConfigKeyError
import re
from omegaconf import OmegaConf
import numpy as np
import torch
from typing import Dict
from lib.utils import create_jinja2_env, stack_tensors_in_dict, convert_list

jinja2_env = create_jinja2_env()

class Interface(HttpNIMApiInterface):
    def __init__(self):
        super().__init__()
        self._inference = GenericInference()
        self._inference.initialize()

    def process_request(
            self,
            request: data_model.{{ request_class }},
    ):
        result = json.loads(request.model_dump_json())
        # load the input config if there is any
        input_config = None
        try:
            input_config = OmegaConf.to_container(global_config.server.endpoints.infer.requests)
        except ConfigKeyError:
            pass
        if input_config is not None:
            req_tpl = input_config.pop("{{ request_class }}", None)
            if req_tpl is not None:
                req_tpl = base64.b64decode(req_tpl).decode()
                json_string = jinja2_env.from_string(req_tpl).render(request=result)
                result = json.loads(json_string)
            for key, value in input_config.items():
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

    def process_response(
        self,
        request: data_model.{{ request_class }},
        response: Dict[str, np.ndarray],
    ) -> data_model.{{ response_class }}:
        self.logger.debug(f"Processing response {response}")
        type_map = { i.name: i.data_type for i in global_config.output}

        # load the input config if there is any
        output_config = None
        try:
            output_config = global_config.server.endpoints.infer.responses
        except ConfigKeyError:
            pass
        tpl = base64.b64decode(output_config["{{ response_class }}"]).decode()

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
        json_string = jinja2_env.from_string(tpl).render(request=request, response=response)
        self.logger.debug(f"Sending json payload: {json_string}")
        return data_model.{{ response_class }}(**json.loads(json_string))

    @HttpNIMApiInterface.route("{{ endpoints.infer }}", methods=["post"])
    async def infer(
        self,
        request: Request,
        body: data_model.{{ request_class }}
    ) -> data_model.{{ response_class }}:
        self.logger.info("infer called")
        in_data = self.process_request(body)
        self.logger.debug(f"request processed as {in_data}")
        result =  await self._inference.execute(in_data)
        response = self.process_response(request, result)
        self.logger.debug(f"response generated as {response}")
        return response

def main():
    interface = Interface()
    try:
        interface.serve()
    except Exception as e:
        interface.logger.exception(e)

if __name__ == "__main__":
    main()