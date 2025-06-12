import time
import argparse
from openai import OpenAI


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-url", type=str, help="Image URL to send for inference", default=None)
    parser.add_argument("--video-path", type=str, help="Video path to send for inference", default=None)
    args = parser.parse_args()

    image_url = args.image_url if args.image_url else ""
    video_path = args.video_path if args.video_path else ""

    if image_url:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe what you see in this image."
                    },
                    {
                        "type": "image",
                        "image": f"{image_url}"
                    }
                ]
            }
        ]
    elif video_path:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe what you see in this video."
                    },
                    {
                        "type": "video",
                        "video": f"{video_path}"
                    }
                ]
            }
        ]

    client = OpenAI(base_url="http://0.0.0.0:8803/v1", api_key="not-used")
    start_time = time.time()
    chat_response = client.chat.completions.create(
        model="nvidia/vila",
        messages=messages,
        max_tokens=512,
        stream=False
    )
    infer_time = time.time() - start_time
    print(f"Inference time: {infer_time} seconds")
    assistant_message = chat_response.choices[0].message
    print(assistant_message)