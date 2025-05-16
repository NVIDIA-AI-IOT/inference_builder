import transformers
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch
import os
import json

class QwenTokenizer:
    name = "qwen-tokenizer"

    def __init__(self, config):
        self._is_encoder = config["is_encoder"]
        self._im_start = "<|im_start|>"
        self._im_end = "<|im_end|>"
        if not self._is_encoder:
            self._skip_special_tokens = config["skip_special_tokens"]
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            config["tokenizer_path"], legacy=False, trust_remote_code=True)
        self.image_start_id = 151857

    def __call__(self, *args):
        if self._is_encoder:
            # tokenizer sample with prompt tuning
            im_start_tokens = [self._tokenizer.im_start_id]
            im_end_tokens = [self._tokenizer.im_end_id]
            nl_tokens = self._tokenizer.encode("\n")
            system_tokens = self._tokenizer.encode("system") + nl_tokens + self._tokenizer.encode("You are a helpful assistant.") + nl_tokens
            system_tokens = im_start_tokens + system_tokens + im_end_tokens + nl_tokens
            msg_strs = [ v.decode() for v in args[0] ]
            user_prompt_tokens = []
            for msg in msg_strs:
                user_prompt_tokens += self._tokenizer.encode("user") + nl_tokens + self._tokenizer.encode(msg) + nl_tokens
            user_prompt_tokens = system_tokens + im_start_tokens + user_prompt_tokens + im_end_tokens
            output = user_prompt_tokens + im_start_tokens + self._tokenizer.encode("assistant") + nl_tokens
            return torch.tensor(output, dtype=torch.int32)
        else:
            ids_list = args[0]
            text_list = []
            for ids in ids_list:
                bos_pos = np.where(ids == self.image_start_id)[0].tolist()
                eos_pos = np.where(ids == self.image_start_id + 1)[0].tolist()
                ids = np.concatenate([ids[:bos_pos[0]], ids[eos_pos[0]:]])
                s = self._tokenizer.decode(ids, skip_special_tokens=self._skip_special_tokens)
                # convert it to bytes to avoid error from triton on unicode characters
                text_list.append(s.encode('utf-8'))
            return np.array(text_list)

class VisualProcessor:
    name = "qwen-visual-processor"
    def __init__(self, config):
        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.std = (0.26862954, 0.26130258, 0.27577711)
        self.image_size = config.get("image_size", 448)
        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def __call__(self, *args):
        image = args[0].permute(2, 0, 1)
        return self.image_transform(image)

class QwenVisionMuxer:
    name = "qwen-vision-muxer"
    def __init__(self, config):
        model_home = config["model_home"]
        self.device_id = config["device_id"]
        llm_config_path = os.path.join(model_home, "config.json")
        with open(llm_config_path, "r") as f:
            pretrained_config = json.load(f)
            self.vocab_size = pretrained_config["pretrained_config"]["vocab_size"]
        self.image_start_id = 151857

    def __call__(self, input_ids, features):
        bos_pos = torch.where(input_ids == self.image_start_id)
        eos_pos = torch.where(input_ids == self.image_start_id + 1)
        fake_prompt_id = torch.arange(
            self.vocab_size,
            self.vocab_size + features.shape[0],
            device="cuda",
        )
        input_ids[bos_pos[0] + 1:eos_pos[0]] = fake_prompt_id
        input_lengths = torch.tensor([input_ids.shape[0]],
                                     dtype=torch.uint32).cuda()
        vocab_size = torch.tensor([features.shape[0]],
                                 dtype=torch.uint32).cuda()
        return input_ids, input_lengths, vocab_size, features