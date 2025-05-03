import numpy as np

class DummyBackend(ModelBackend):
    """Python TensorRT Backend from polygraph"""
    def __init__(self, model_config:Dict, model_home: str, device_id: int=0):
        super().__init__(model_config, model_home, device_id)
        self._model_name = model_config["name"]
        self._input_config = [i for i in model_config['input']]
        self._output_config = [o for o in model_config['output']]

        logger.info(f"DummyBackend created for {self._model_name} to generate {[o['name'] for o in self._output_config]}")

    def __call__(self, *args, **kwargs):
        logger.debug(f"DummyBackend {self._model_name} triggerred with  {args if args else kwargs}")
        in_data = list(args) if args else [kwargs]
        out_data = []
        for data in in_data:
            dummy_data = {}
            # validate the input data
            for name, tensor in data.items():
                if name not in [i['name'] for i in self._input_config]:
                    raise ValueError(f"Input {name} not found in the input config")
                if tensor.shape != tuple([i['dims'] for i in self._input_config if i['name'] == name][0]):
                    raise ValueError(f"Input {name} has invalid shape")
            # generate the output data
            for config in self._output_config:
                name = config['name']
                data = np.random.randn(*config['dims'])
                dummy_data[name] = data
            out_data.append(dummy_data)
        # generate a batch of dummy data
        yield out_data
