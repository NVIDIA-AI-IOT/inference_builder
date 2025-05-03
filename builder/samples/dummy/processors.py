import numpy as np

class DummyPreprocessor:
    name = "dummy-preprocessor"
    def __init__(self, config):
        self.network_size = config['network_size']

    def __call__(self, *args, **kwargs):
        return np.random.randn(*self.network_size)


class DummyTokenizer:
    name = "dummy-tokenizer"
    def __init__(self, config):
        pass

    def __call__(self, *args, **kwargs):
        return np.random.randint(0, 100, size=(1, 10)), np.array([10])
