

import argparse
import tempfile
import shutil
from omegaconf import OmegaConf
import cookiecutter.main
import cookiecutter
import logging
from pathlib import Path
import datamodel_code_generator as data_generator
from utils import get_resource_path, copy_files
from triton.model_config_pb2 import ModelConfig, ModelInput, ModelOutput
from triton.utils import datatype_mapping
from google.protobuf import text_format

ALLOWED_SERVER = ["triton"]

logging.basicConfig(level=logging.INFO)

def build_args(parser):
    parser.add_argument(
        "--server-type",
        type=str,
        nargs='?',
        default='triton',
        choices=ALLOWED_SERVER,
        help="Choose the server type"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        nargs='?',
        default='.',
        help="Output directory"
    )
    parser.add_argument(
        "--api-spec",
        type=argparse.FileType('r'),
        nargs='?',
        help="File for OpenAPI specification"
    )
    parser.add_argument("config", type=str, help="Path the the configuration")

def build_tree(server_type, config, temp_dir):
    configuration = OmegaConf.to_yaml(config)
    ep = OmegaConf.to_container(config.server.endpoint_map)
    cookiecutter.main.cookiecutter(
        get_resource_path(f"builder/boilerplates/{server_type}"),
        no_input=True,
        extra_context={"service_name" : config.name, "configuration": '{% raw %}' + configuration +'{% endraw %}', "endpoints": ep},
        output_dir=temp_dir)
    return Path(temp_dir) / Path(config.name)

def build_inference(server_type, config, model_repo_dir):
    if server_type == "triton":
        for model in config.models:
            if model.backend == "tensorrtllm":
                model_config = ModelConfig()
                for key, value in model.items():
                    if hasattr(model_config, key):
                        if key == "input":
                            for i in value:
                                model_input = model_config.input.add()
                                for k,v in i.items():
                                    if hasattr(model_input, k):
                                        if k == "data_type":
                                            setattr(model_input, k, datatype_mapping[v])
                                        elif k == "dims":
                                            for dim in v:
                                                model_input.dims.append(dim)
                                        else:
                                            setattr(model_input, k, v)
                        elif key == "output":
                            for i in value:
                                model_output = ModelOutput()
                                for k,v in i.items():
                                    if hasattr(model_output, k):
                                        if k == "data_type":
                                            setattr(model_output, k, datatype_mapping[v])
                                        elif k == "dims":
                                            for dim in v:
                                                model_output.dims.append(dim)
                                        else:
                                            setattr(model_output, k, v)
                                model_config.output.append(model_output)
                        else:
                            setattr(model_config, key, value)
                pbtxt_str = text_format.MessageToString(model_config)
                (model_repo_dir/f"{model.name}/1").mkdir(parents=True)
                pbtxt_path = model_repo_dir/model.name/"config.pbtxt"
                with open(pbtxt_path, 'w') as f:
                    f.write(pbtxt_str)
    else:
        raise Exception("Not implemented")



def build_server(server_type, api_schema, config, output_dir):
    # generate pydantic data models from swagger spec
    output_file = output_dir / "data_model.py"
    data_generator.generate(
        api_schema, output=output_file, output_model_type=data_generator.DataModelType.PydanticV2BaseModel
    )

def main(args):
    api_schema = None
    with args.api_spec as f:
        api_schema = f.read()
    config = OmegaConf.load(args.config)
    with tempfile.TemporaryDirectory() as temp_dir:
        tree = build_tree(args.server_type, config, temp_dir)
        build_server(args.server_type, api_schema, config, tree/"server/optimized")
        build_inference(args.server_type, config.inference, tree/"server/optimized/model_repo")
        common_src = get_resource_path("templates/common")
        copy_files(common_src, tree/"server/optimized/common")
        target = Path(args.output_dir).resolve() / config.name
        try:
            shutil.copytree(tree, target, dirs_exist_ok=True)
        except FileExistsError:
            logging.error(f"{target} already exists in the output directory")




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference Builder")
    build_args(parser)
    main(parser.parse_args())