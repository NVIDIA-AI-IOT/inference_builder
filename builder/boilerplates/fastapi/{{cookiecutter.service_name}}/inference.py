
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
    SYMLINK = os.path.join(CACHE_PATH, "model-repo/tao")
    
    # Remove existing symlink if it exists
    if os.path.islink(SYMLINK):
        os.remove(SYMLINK)
        print(f"Symlink deleted: {SYMLINK}")
    
    # # Create workspace directory if it doesn't exist
    # os.makedirs(MODEL_PATH, exist_ok=True)

    # Create new symlink
    os.symlink(MODEL_PATH, SYMLINK)
    print(f"Created symlink: {SYMLINK} -> {MODEL_PATH}")

def main():
    setup_tao_config_symlink()
    update_model_config_paths()
    server.inference.main()