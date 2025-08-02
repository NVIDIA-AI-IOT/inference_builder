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

"""
This file exists to match nimlib's start_server executable requirements.
The start_server will call:
    @check_max_gpu_utilization
    def start_inference_server():
        import inference
        inference.main()

Therefore:
1. The file must be named 'inference.py'
2. Must expose a main() method
3. Must be in the Python path (here: /opt/nim/)
"""
import os
import fileinput
import server.inference
from nimlib.env import get_workspace_from_env
MODEL_PATH = get_workspace_from_env()

def update_model_config_paths():
    config_file = os.path.join(MODEL_PATH, "nvdsinfer_config.yaml")
    if os.path.exists(config_file):
        with fileinput.input(config_file, inplace=True) as f:
            for line in f:
                line = line.replace('onnx-file: ./', f'onnx-file: {MODEL_PATH}/')
                line = line.replace('model-engine-file: ./', f'model-engine-file: {MODEL_PATH}/')
                line = line.replace('labelfile-path: ./', f'labelfile-path: {MODEL_PATH}/')
                print(line, end='')

def setup_tao_config_symlink():
    CACHE_PATH = os.getenv('NIM_CACHE_PATH')
    MODEL_REPO_PATH = os.path.join(CACHE_PATH, "model-repo")
    SYMLINK = os.path.join(MODEL_REPO_PATH, "tao")

    # Remove existing symlink if it exists
    if os.path.islink(SYMLINK):
        os.remove(SYMLINK)
        print(f"Symlink deleted: {SYMLINK}")

    # Create model-repo directory if it doesn't exist
    # when creating a symlink with os.symlink(src, dst),
    # the parent directory of the destination path (dst) must exist,
    # but the destination path itself should not exist
    os.makedirs(MODEL_REPO_PATH, exist_ok=True)
    print(f"Created directory: {MODEL_REPO_PATH}")

    # Create new symlink
    os.symlink(MODEL_PATH, SYMLINK)
    print(f"Created symlink: {SYMLINK} -> {MODEL_PATH}")

def main():
    print("Run inference.py by nimlib's start_server import")
    setup_tao_config_symlink()
    update_model_config_paths()
    server.inference.main()

if __name__ == "__main__":
    print("Directly run inference.py as main entrypoint")
    server.inference.main()
