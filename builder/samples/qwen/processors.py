from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

class QwenVLProcessor:
    name = "qwen-vl-processor"
    def __init__(self, config):
        model_home = config["model_home"]
        self._processor = AutoProcessor.from_pretrained(model_home)

    def __call__(self, *args):
        messages = args[0]
        max_new_tokens = args[1][0]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        return inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"], inputs["image_grid_thw"], max_new_tokens, inputs["input_ids"]

class QwenVLTokenizer:
    name = "qwen-vl-tokenizer"
    def __init__(self, config):
        self._processor = AutoProcessor.from_pretrained(config["model_home"])

    def __call__(self, *args):
        generated_ids = args[0]
        input_ids = args[1]
        print(f"generated_ids: {generated_ids}")
        print(f"input_ids: {input_ids}")
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids.unsqueeze(0))
        ]
        return self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )