from transformers import AutoProcessor
import torch
import numpy as np

class QwenVLProcessor:
    name = "qwen-vl-processor"
    def __init__(self, config):
        from qwen_vl_utils import process_vision_info
        model_home = config["model_home"]
        self._processor = AutoProcessor.from_pretrained(model_home)
        self._process_vision_info = process_vision_info

    def __call__(self, *args):
        messages = args[0]
        max_new_tokens = args[1]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        if "pixel_values" in inputs:
            return inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"], inputs["image_grid_thw"], None, None, max_new_tokens, inputs["input_ids"]
        else:
            return inputs["input_ids"], inputs["attention_mask"], None, None, inputs["pixel_values_videos"], inputs["video_grid_thw"], max_new_tokens, inputs["input_ids"]

    def apply_chat_template(self, prompt, multimodal_data):
        # Build content list with media placeholders
        content = []

        # Add one placeholder for each media item
        for media_type, items in multimodal_data.items():
            for _ in items:
                content.append({"type": media_type})

        # Add the text content
        content.append({
            "type": "text",
            "text": prompt
        })

        conversation = [{"role": "user", "content": content}]
        return self._processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

class QwenVLVideoProcessor(QwenVLProcessor):
    name = "qwen-vl-video-processor"
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, *args):
        # Convert from DLPack to torch tensor, then transform from HWC to CHW and normalize
        prompts = args[0]
        videos = args[1]
        if isinstance(prompts, np.ndarray):
            prompts = prompts.tolist()
        elif not isinstance(prompts, list):
            prompts = [prompts]
            videos = [videos]

        inputs = []
        for prompt, frames in zip(prompts, videos):
            tensors = [
                torch.utils.dlpack.from_dlpack(i).permute(2, 0, 1).float() / 255.0
                for i in frames
            ]
            multimodal_data = {"video": [tensors]}
            inputs.append({
                "prompt": self.apply_chat_template(prompt, multimodal_data),
                "multi_modal_data": multimodal_data
            })
        return inputs


class QwenVLImageProcessor(QwenVLProcessor):
    name = "qwen-vl-image-processor"
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, *args):
        # Convert from DLPack to torch tensor, then transform from HWC to CHW and normalize
        prompts = args[0]
        images = args[1]
        if isinstance(prompts, np.ndarray):
            prompts = prompts.tolist()
        elif not isinstance(prompts, list):
            prompts = [prompts]
            images = [images]

        inputs = []
        for prompt, image in zip(prompts, images):
            tensors = [image.permute(2, 0, 1).float() / 255.0]
            multimodal_data = {"image": tensors}
            inputs.append({
                "prompt": self.apply_chat_template(prompt, multimodal_data),
                "multi_modal_data": multimodal_data
            })
        return inputs


class QwenVLImageLoader(QwenVLProcessor):
    name = "qwen-vl-image-loader"
    def __init__(self, config):
        super().__init__(config)
        from tensorrt_llm.inputs import default_image_loader
        self._default_image_loader = default_image_loader
        self._model_home = config["model_home"]

    def __call__(self, *args):
        prompts = args[0].tolist() if isinstance(args[0], np.ndarray) else args[0]
        images = args[1].tolist() if isinstance(args[1], np.ndarray) else args[1]
        assert len(images) == len(prompts)
        inputs = self._default_image_loader(prompts, images)
        for i in inputs:
            i["prompt"] = self.apply_chat_template(i["prompt"], i["multi_modal_data"])
        return inputs

class QwenVLVideoLoader(QwenVLProcessor):
    name = "qwen-vl-video-loader"
    def __init__(self, config):
        super().__init__(config)
        from tensorrt_llm.inputs import default_video_loader
        self._default_video_loader = default_video_loader
        self._model_home = config["model_home"]
        self._num_frames = config.get("num_frames", 8)

    def __call__(self, *args):
        prompts = args[0]
        if isinstance(prompts, np.ndarray):
            prompts = prompts.tolist()
        elif not isinstance(prompts, list):
            prompts = [prompts]
        videos = args[1]
        if isinstance(videos, np.ndarray):
            videos = videos.tolist()
        elif not isinstance(videos, list):
            videos = [videos]
        assert len(videos) == len(prompts)
        inputs = self._default_video_loader(prompts, videos, num_frames=self._num_frames)
        for i in inputs:
            i["prompt"] = self.apply_chat_template(i["prompt"], i["multi_modal_data"])
        return inputs


class QwenVLTokenizer:
    name = "qwen-vl-tokenizer"
    def __init__(self, config):
        self._processor = AutoProcessor.from_pretrained(config["model_home"])

    def __call__(self, *args):
        generated_ids = args[0]
        input_ids = args[1]
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids.unsqueeze(0))
        ]
        return self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )