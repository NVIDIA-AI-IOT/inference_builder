import torch
import numpy as np

class OpenclipTokenizer:
    name = "openclip-tokenizer"
    def __init__(self, config):
        import open_clip
        open_clip.add_model_config(config["model_path"])
        self._tokenizer = open_clip.get_tokenizer("NVCLIP_224_700M_ViTH14")

    def __call__(self, *args, **kwargs):
        strs = [s.decode("utf-8") for s in args[0]]
        return self._tokenizer(strs)

class VisionPreprocessor:
    name = "nvclip-vision-preprocessor"
    def __init__(self, config):

        from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
        self._transform = Compose(
            [
                Resize((224, 224)),
                CenterCrop(224),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __call__(self, *args, **kwargs):
        image = args[0].to(torch.float32).permute(2, 0, 1)
        return self._transform(image)

class NvClipPostProcessor:
    name = "nvclip-postprocessor"
    def __init__(self, config):
        self._config = config

    def __call__(self, *args, **kwargs):
        text = args[0].tolist()
        images = args[1].tolist()
        total_tokens = sum(len(s) for s in text)
        num_images = len(images)
        indices = [s.decode("utf-8") for s in args[2]]
        embeddings = []
        for index in indices:
            if index == "text":
                embeddings.append(text.pop(0))
            elif index == "image":
                embeddings.append(images.pop(0))
        return np.array(embeddings), np.array([total_tokens]), np.array([num_images])


