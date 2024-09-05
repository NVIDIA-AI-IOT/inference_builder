

import argparse
import base64
import tempfile
import shutil
from omegaconf import OmegaConf
import cookiecutter.main
import cookiecutter
import logging
from typing import Dict
from pathlib import Path
import datamodel_code_generator as data_generator
import semver
from utils import get_resource_path, copy_files
from triton.utils import generate_pbtxt
from omegaconf.errors import ConfigKeyError
from jinja2 import Environment, FileSystemLoader

ALLOWED_SERVER = ["triton"]

logging.basicConfig(level=logging.INFO)
OmegaConf.register_new_resolver("multiline", lambda x: x, replace=False)

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
    def encode_templates(templates):
        encoded_templates = dict()
        for key, value in templates.items():
            # these are json templates and need be encoded before being embeded as yaml strings
            encoded_templates[key] = base64.b64encode(value.encode())
        return encoded_templates

    input_templates = None
    output_templates = None
    try:
        input_templates = config.io_map.input.templates
        output_templates = config.io_map.output.templates
    except ConfigKeyError:
        pass
    if input_templates:
        config.io_map.input.templates = encode_templates(input_templates)
    if output_templates:
        config.io_map.output.templates = encode_templates(output_templates)
    configuration = OmegaConf.to_yaml(config)
    ep = OmegaConf.to_container(config.endpoints)
    cookiecutter.main.cookiecutter(
        get_resource_path(f"builder/boilerplates/{server_type}"),
        no_input=True,
        extra_context={"service_name" : config.name, "configuration": '{% raw %}' + configuration +'{% endraw %}', "endpoints": ep},
        output_dir=temp_dir)
    return Path(temp_dir) / Path(config.name)

def build_inference(server_type, config, model_repo_dir: Path):
    env = dict()
    tpl_dir = get_resource_path("templates")
    jinja_env = Environment(loader=FileSystemLoader(tpl_dir))
    if hasattr(config, "environment"):
        env = OmegaConf.to_container(config.environment)
    if server_type == "triton":
        for model in config.models:
            fallback = False
            # generate the pbtxt for the tensorrtllm backend
            backend = model.backend.split('-')
            if len(backend) == 2 and backend[0] in env:
                required_version = backend[1]
                env_version = env[backend[0]]
                if semver.compare(env_version, required_version) < 0:
                    fallback = True
                    model.backend = backend[0]
            pbtxt_str = generate_pbtxt(OmegaConf.to_container(model), fallback)
            (model_repo_dir/f"{model.name}/1").mkdir(parents=True)
            pbtxt_path = model_repo_dir/model.name/"config.pbtxt"
            with open(pbtxt_path, 'w') as f:
                f.write(pbtxt_str)
            if fallback:
                target_dir = model_repo_dir/f"{model.name}/1"
                model_file = target_dir/"model.py"
                triton_tpl = jinja_env.get_template('triton/model.jinja.py')
                if "tensorrt" in model.backend:
                    trt_tpl = get_resource_path("templates/trt/backend.py")
                    with open(trt_tpl, 'r') as f:
                        trt_backend = f.read()
                        output = triton_tpl.render(backend=trt_backend)
                        with open (model_file, 'w') as o:
                            o.write(output)
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
        build_inference(args.server_type, config, tree/"server/optimized/model_repo")
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