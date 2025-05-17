import importlib

class PytorchBackend(ModelBackend):
    """Pytorch Backend to run models from Huggingface"""
    def __init__(self, model_config:Dict, model_home: str, device_id: int=0):
        super().__init__(model_config, model_home, device_id)
        self._model_name = model_config["name"]
        self._output_names = [o["name"] for o in model_config["output"]]
        self._model_class = model_config["parameters"].get("model_class", "AutoModelForCausalLM")
        self._module = importlib.import_module("transformers")
        model_class = getattr(self._module, self._model_class)
        logger.info(f"Loading pre-trained model {self._model_name} of type {self._model_class} from {self._model_home}")
        self._model = model_class.from_pretrained(self._model_home, torch_dtype="auto", device_map="auto")
        logger.info(f"Model {self._model_name} loaded from {self._model_home}")

    def __call__(self, *args, **kwargs):
        in_data_list = args if args else [kwargs]
        for in_data in in_data_list:
            result = self._model.generate(**in_data)
            if len(self._output_names) != len(result):
                raise ValueError(f"Number of output names ({len(self._output_names)}) does not match the number of output tensors ({len(result)})")
            for i, tensor in enumerate(result):
                yield {self._output_names[i]: tensor}
