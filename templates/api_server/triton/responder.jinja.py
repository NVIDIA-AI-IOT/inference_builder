import json
from .data_model import {{ triton.request_class }}, {{ triton.response_class }}, {{ triton.streaming_response_class }}
from config import global_config
from lib.utils import create_jinja2_env, convert_list
from typing import Dict, Any, Optional, List, Union
import numpy as np
import torch
from typing import Dict, Any
from lib.responder import ResponderBase
from inferencemodeltoolkit.interfaces.fastapi.triton import (
    TritonInferenceHandler,
    TritonPydanticValidator,
)

{% for responder in responders %}
{% if responder.name == "infer" %}
infer_operation = "{{responder.operation}}"
{% endif %}
{% endfor %}

jinja2_env = create_jinja2_env()

class TritonInferenceHandlerBridge(TritonInferenceHandler):
    def __init__(self, responder):
        super().__init__(
             triton_url="grpc://localhost:8001",
             model_name="{{ service_name }}",
             stream_triton=True,
             stream_http=False)
        self._responder = responder

    def process_request(
            self,
            request: {{ triton.request_class }},
            headers: Dict[str, str]
    ):
        result = self._responder.process_request("infer", request)
        # Triton inference handler transforms the values to numpy arrays based on the model metadata
        return result

    def process_response(
        self,
        request: {{ triton.request_class }},
        response: Dict[str, np.ndarray],
        previous_responses: Optional[List[Dict[str, np.ndarray]]],
        headers
    ) -> Union[{{ triton.response_class }}, {{ triton.streaming_response_class }}]:
        logger.debug(f"Processing response {response}")
        type_map = { i.name: i.data_type for i in global_config.output}
        # Formulating streaming response
        if hasattr(request, 'stream') and request.stream:
            streamed = dict()
            for name, value in response.items():
                if value is None:
                    logger.error(f"{name} in response is None")
                    continue
                expected_type = type_map[name]
                if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                    l = value.tolist()
                    if expected_type == "TYPE_STRING" and value.dtype != np.string_:
                        l = [i.decode("utf-8", "ignore") for i in l]
                    streamed[name] = l
                else:
                    streamed[name] = value
            json_string = self._responder.process_streamed_response("infer", request, streamed)
            return {{ triton.streaming_response_class }}(**json.loads(json_string))

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
                    acc[name] = convert_list(l, lambda i: i.decode("utf-8", "ignore"))
                elif len(acc[name].shape) == 1 and len(l) == 1:
                    acc[name] = l[0]
                else:
                    acc[name] = l
        json_string = self._responder.process_response("infer", request, acc)
        return {{ triton.response_class }}(**json.loads(json_string))

class TritonResponder(ResponderBase):
    def __init__(self, operations, app):
        super().__init__()
        self._inference = TritonInferenceHandlerBridge()
        validator = TritonPydanticValidator(self)
        _, additional_responses = validator._validate_pydantic_hints(self._inference)
        # override the infer operation
        app.post(
            operations[infer_operation],
            response_model_exclude_none=True,
            responses=additional_responses,
        )(self._inference._infer)
        # initialize the action map for the other operations
        {% for responder in responders %}
        {% if responder.name != "infer"  %}
        self._action_map["{{ responder.operation }}"] = self.{{ responder.name }}
        {% endif %}
        {% endfor %}

{% for responder in responders %}
{% if responder.name != "infer" %}
{{ responder.implementation }}
{% endif %}
{% endfor %}