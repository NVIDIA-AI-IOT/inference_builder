import base64
from io import BytesIO
import argparse
from openai import OpenAI


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="File to send for inference", default=None)
    args = parser.parse_args()

    with open(args.file, "rb") as f:
        fetched_bytes = BytesIO(f.read())
        fetched_b64 = base64.b64encode(fetched_bytes.getvalue()).decode()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe what you see in this image.<img></img>"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{fetched_b64}"
                    }
                }
            ]
        }
    ]
    client = OpenAI(base_url="http://0.0.0.0:8803/v1", api_key="not-used")
    chat_response = client.chat.completions.create(
        model="nvidia/vila",
        messages=messages,
        max_tokens=512,
        stream=False
    )
    assistant_message = chat_response.choices[0].message
    print(assistant_message)