import os, json
import numpy as np

class VilaMuxer:
    name = "vila-muxer"
    def __init__(self, config):
        model_path = config["model_path"]
        llm_config_path = os.path.join(model_path, "fp16", "1-gpu", "config.json")
        with open(llm_config_path, "r") as f:
            config = json.load(f)
            self.vocab_size = config["pretrained_config"]["vocab_size"]

    def __call__(self, input_ids, features):
        image_embed_input_ids = np.arange(
                self.vocab_size,
                self.vocab_size + features.shape[0])
        extended_ids = np.append(input_ids, image_embed_input_ids)
        id_length = extended_ids.shape[0]
        vocab_size = features.shape[0]
        return extended_ids, np.array([id_length]), np.array([vocab_size]), features

class VilaVisionEncoderProcessor:
    name = "vila-preprocessor"
    def __init__(self, config):
        model_path = config["model_path"]
        from llava.model.multimodal_encoder.siglip.image_processing_siglip import SiglipImageProcessor
        self._preprocessor = SiglipImageProcessor.from_pretrained(model_path)

    def __call__(self, *args, **kwargs):
        return self._preprocessor(*args)['pixel_values'][0],