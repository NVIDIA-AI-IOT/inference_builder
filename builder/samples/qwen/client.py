# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import argparse
from openai import OpenAI
import base64
import mimetypes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="Image paths to send for inference", default=None, nargs="*")
    parser.add_argument("--videos", type=str, help="Video paths to send for inference", default=None, nargs="*")
    parser.add_argument("--endpoint", type=str, help="Endpoint to send for inference", default="http://0.0.0.0:8800/v1", nargs="?")
    args = parser.parse_args()

    messages = []
    if args.images:
        for image in args.images:
            with open(image, "rb") as f:
                data = base64.b64encode(f.read()).decode()
                mime_type = mimetypes.guess_type(image)[0]
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please describe the image in detail."
                        },
                        {
                            "type": "image",
                            "image": f"data:{mime_type};base64,{data}"
                        }
                    ]
                })
    elif args.videos:
        for video in args.videos:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please describe the video in detail."
                    },
                    {
                        "type": "video",
                        "video": f"{video}"
                    }
                ]
            })
    else:
        raise ValueError("No images or videos provided")

    client = OpenAI(base_url=args.endpoint, api_key="not-used")
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