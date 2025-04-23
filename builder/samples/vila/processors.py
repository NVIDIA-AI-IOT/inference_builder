import os, json
import numpy as np
import transformers

class VilaTokenizer:
    name = "vila-tokenizer"
    def __init__(self, config):
        self._is_encoder = config["is_encoder"]
        if not self._is_encoder:
            self._skip_special_tokens = config["skip_special_tokens"]
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(config["model_path"])

    def __call__(self, *args):
        if self._is_encoder:
            msg_strs = [ v.decode() for v in args[0] ]
            output = self._tokenizer(*msg_strs)
            return np.array(output["input_ids"])
        else:
            ids_list = args[0]
            text_list = []
            for ids in ids_list:
                text_list.append(self._tokenizer.decode(ids, skip_special_tokens=self._skip_special_tokens))
            return np.array(text_list, dtype=np.string_)

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