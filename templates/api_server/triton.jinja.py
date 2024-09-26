import base64
import json
from typing import Dict, Optional, List, Union
import numpy as np
from inferencemodeltoolkit.interfaces.fastapi import FastApiTritonInterface
import data_model
from lib.utils import get_logger
from config import global_config
from omegaconf.errors import ConfigKeyError
from jinja2 import Environment
import re
from omegaconf import OmegaConf

logger = get_logger(__name__)

def start_with(field, s):
    return field.startswith(s)

def remove_prefix(value, prefix):
    if value.startswith(prefix):
        return value[len(prefix):]
    return value

jinja2_env = Environment()
jinja2_env.tests["startswith"] = start_with
jinja2_env.filters["remove_prefix"] = remove_prefix

class Interface(FastApiTritonInterface):
    def process_request(
            self,
            request: data_model.{{ request_class }},
            headers: Dict[str, str]
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
        previous_responses: Optional[List[Dict[str, np.ndarray]]],
        headers
    ) -> Union[data_model.{{ response_class }}, data_model.{{ streaming_response_class }}]:
        logger.debug(f"Processing response {response}")
        type_map = { i.name: i.data_type for i in global_config.output}

        # load the input config if there is any
        output_config = None
        try:
            output_config = global_config.server.endpoints.infer.responses
        except ConfigKeyError:
            pass
        tpl = base64.b64decode(output_config["{{ response_class }}"]).decode()
        stpl = base64.b64decode(output_config["{{ streaming_response_class }}"]).decode()

        # Formulating streaming response
        if hasattr(request, 'stream') and request.stream:
            streamed = dict()
            for name, value in response.items():
                if value is None:
                    logger.error(f"{name} in response is None")
                    continue
                expected_type = type_map[name]
                if isinstance(value, np.ndarray):
                    l = value.tolist()
                    if expected_type == "TYPE_STRING" and value.dtype != np.string_:
                        l = [i.decode("utf-8", "ignore") for i in l]
                    streamed[name] = l
                else:
                    streamed[name] = value
            json_string = jinja2_env.from_string(stpl).render(request=request, response=streamed)
            return data_model.{{ streaming_response_class }}(**json.loads(json_string))

        # Formulating aggregated response from all the responses
        responses = previous_responses + [response]  if previous_responses else [response]
        acc = dict()
        # aggregate the data
        for response in responses:
            for name, value in response.items():
                if value is None:
                    logger.error(f"{name} in response is None")
                    continue
                if isinstance(value, np.ndarray):
                    if name in acc:
                        acc[name] = np.append(acc[name], value)
                    else:
                        acc[name] = value
                else:
                    acc[name] = acc[name] + value if name in acc else value
        # transform numpy ndarray to universal value types
        for name in acc:
            expected_type = type_map[name]
            if isinstance(acc[name], np.ndarray):
                l = acc[name].tolist()
                if acc[name].dtype != np.string_ and expected_type == "TYPE_STRING":
                    acc[name] = ' '.join([i.decode("utf-8", "ignore") for i in l])
                elif len(acc[name].shape) == 1 and len(l) == 1:
                    acc[name] = l[0]
                else:
                    acc[name] = l
        json_string = jinja2_env.from_string(tpl).render(request=request, response=acc)
        return data_model.{{ response_class }}(**json.loads(json_string))

def main():
    interface = Interface(
        triton_url="grpc://localhost:8001",
        model_name="{{ service_name }}",
        stream_triton=True,
        stream_http=lambda request: request.stream if hasattr(request, 'stream') else False,
        infer_endpoint="{{ endpoints.infer }}",
        health_endpoint="{{ endpoints.health}}",
        triton_timeout_s = 60,
    )
    interface.serve()

if __name__ == "__main__":
    main()