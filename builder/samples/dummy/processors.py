import numpy as np
import torch

class DummyPreprocessor:
    name = "dummy-preprocessor"
    def __init__(self, config):
        self.network_size = config['network_size']

    def __call__(self, *args, **kwargs):
        if len(args) != 1:
            raise ValueError("DummyPreprocessor expects exactly one argument")
        input = args[0]
        if len(input.shape) != 3:
            raise ValueError("DummyPreprocessor expects a 3D tensor")
        if isinstance(input, np.ndarray) and input.dtype != np.uint8:
            raise ValueError("DummyPreprocessor expects a uint8 tensor")
        if isinstance(input, torch.Tensor) and input.dtype != torch.uint8:
            raise ValueError("DummyPreprocessor expects a uint8 tensor")
        return np.random.randn(*self.network_size)


class DummyTokenizer:
    name = "dummy-tokenizer"
    def __init__(self, config):
        pass

    def __call__(self, *args, **kwargs):
        return np.random.randint(0, 100, size=10), np.array([10])
