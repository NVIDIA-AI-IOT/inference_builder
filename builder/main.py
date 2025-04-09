import argparse
import base64
import tempfile
import shutil
from omegaconf import OmegaConf
import cookiecutter.main
import cookiecutter
import logging
from typing import Dict, List
from pathlib import Path
import datamodel_code_generator as data_generator
import semver
from utils import get_resource_path, copy_files, create_tar_gz
from triton.utils import generate_pbtxt
from omegaconf.errors import ConfigKeyError
from jinja2 import Environment, FileSystemLoader
import ast
import os
import subprocess
import validate

ALLOWED_SERVER = ["triton", "fastapi"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")
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
        "-o",
        "--output-dir",
        type=str,
        nargs='?',
        default='.',
        help="Output directory"
    )
    parser.add_argument(
        "-a",
        "--api-spec",
        type=argparse.FileType('r'),
        nargs='?',
        help="File for OpenAPI specification"
    )
    parser.add_argument(
        "-c",
        "--custom-module",
        type=argparse.FileType('r'),
        nargs='*',
        help="Custome python modules"
    )
    parser.add_argument(
        "-x",
        "--exclude-lib",
        action='store_true',
        help="Don't include common lib to the generated code."
    )
    parser.add_argument(
        "-t",
        "--tar-output",
        action='store_true',
        help="Zip the output to a single file"
    )
    parser.add_argument("config", type=str, help="Path the the configuration")
    parser.add_argument(
        "-v",
        "--validation-dir",
        type=str,
        help="valid validation directory path to build validator"
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Use local OpenAPI Generator instead of Docker for OpenAPI client generation")

def build_tree(server_type, config, temp_dir):
    cookiecutter.main.cookiecutter(
        get_resource_path(f"builder/boilerplates/{server_type}"),
        no_input=True,
        extra_context={"service_name": config.name},
        output_dir=temp_dir)
    return Path(temp_dir) / Path(config.name)

def build_custom_modules(custom_modules: List, tree):
    cls_list = []
    tpl_dir = get_resource_path("templates")
    jinja_env = Environment(loader=FileSystemLoader(tpl_dir))
    tree = tree / "custom"
    for m in custom_modules:
        filename = os.path.basename(m.name)
        module_id = os.path.splitext(filename)[0]
        with m as f:
            valid = False
            source = f.read()
            py = ast.parse(source)
            for node in ast.walk(py):
                if isinstance(node, ast.ClassDef):
                    name = None
                    has_call = False
                    for cls_node in node.body:
                        if isinstance(cls_node, ast.Assign):
                            for target in cls_node.targets:
                                if isinstance(target, ast.Name) and target.id == "name":
                                    name = ast.literal_eval(cls_node.value)
                        if isinstance(cls_node, ast.FunctionDef):
                            if cls_node.name == "__call__":
                                has_call = True
                    if name and has_call:
                        if next((i for i in cls_list if i["name"] == name), None):
                            logger.warning(f"Custom class {name} defined more than once")
                            continue
                        cls_list.append({
                            "name": name,
                            "module": module_id,
                            "class_name": node.name
                        })
                        valid = True
            if valid:
                # write the python module
                target_path = tree / f"{module_id}.py"
                with open(target_path, "w") as t:
                    t.write(source)
    custom_tpl = jinja_env.get_template('common/custom.__init__.jinja.py')
    output = custom_tpl.render(classes=cls_list)
    with open(tree/"__init__.py", 'w') as f:
        f.write(output)

def build_inference(server_type, config, output_dir: Path):
    tpl_dir = get_resource_path("templates")
    jinja_env = Environment(loader=FileSystemLoader(tpl_dir))
    triton_model_repo_dir = output_dir/'model_repo'
    triton_tpl = jinja_env.get_template("triton/model.jinja.py")
    generic_tpl = jinja_env.get_template("generic/model.jinja.py")

    # collect the backend: build model.py and pbtxt if the backend is triton
    t_backends = []
    for model in config.models:
        backend_spec = model.backend.split('/')
        if backend_spec[0] == "triton":
            os.makedirs(triton_model_repo_dir/f"{model.name}/1", exist_ok=True)
            if len(backend_spec) < 2:
                raise Exception("Triton backend needs a triton backend type")
            if backend_spec[1] == "python":
                if len(backend_spec) < 3:
                    raise Exception("Triton python backend needs an implementation type")
                # generating triton model for triton backend
                target_dir = triton_model_repo_dir/f"{model.name}/1"
                    # Triton python backend needs a model.py
                backend_tpl = jinja_env.get_template(f"backend/{backend_spec[2]}.jinja.py")
                backend = backend_tpl.render(server_type=server_type)
                output = triton_tpl.render(backends=[backend], top_level=False)
                with open (target_dir/"model.py", 'w') as o:
                    o.write(output)
                if "triton" not in t_backends:
                    t_backends.append("triton")
            # write the pbtxt
            pbtxt_str = generate_pbtxt(OmegaConf.to_container(model), backend_spec[1] )
            pbtxt_path = triton_model_repo_dir/model.name/"config.pbtxt"
            with open(pbtxt_path, 'w') as f:
                f.write(pbtxt_str)
        else:
            bare_backend = backend_spec[0]
            if bare_backend not in t_backends:
                t_backends.append(bare_backend)

    # create backends and model.py
    backends = []
    target_dir = triton_model_repo_dir/f"{config.name}"/"1/" if server_type == "triton" else output_dir / "server"
    for backend in t_backends:
        backend_tpl = jinja_env.get_template(f"backend/{backend}.jinja.py")
        backends.append(backend_tpl.render(server_type=server_type))
        if server_type == "triton":
            # render top level triton backend
            output = triton_tpl.render(backends=backends, top_level=True)
        else:
            # render generic model.y
            output = generic_tpl.render(backends=backends)
        with open (target_dir/"model.py", 'w') as o:
            o.write(output)



def build_server(server_type, model_name, api_spec, config: Dict, output_dir):
    output_dir = output_dir / "server"
    # generate pydantic data models from swagger spec
    tpl_dir = get_resource_path("templates")
    jinja_env = Environment(loader=FileSystemLoader(tpl_dir))
    if server_type == "fastapi":
        fastapi_tpl_dir = get_resource_path("templates/api_server/fastapi/route")
        command = (
            f"fastapi-codegen --input {api_spec.name} --output {output_dir} --output-model-type pydantic_v2.BaseModel "
            f"--template-dir {fastapi_tpl_dir} -m data_model.py --disable-timestamp"
        )
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Failed to generate fastapi data models: {result.stderr}")

        responders = []
        for name, r in config["responders"].items():
            responder = {
                "name": name,
                "operation": r["operation"],
            }
            tpl = jinja_env.get_template(f"responder/{name}.jinja.py")
            responder["implementation"] = tpl.render(**responder)
            responders.append(responder)

        svr_tpl = jinja_env.get_template(f"api_server/{server_type}/responder.jinja.py")
        output = svr_tpl.render(
            service_name=model_name,
            responders=responders
        )
        with open(output_dir/"responder.py", 'w') as f:
            f.write(output)
    else:
        output_file = output_dir / "data_model.py"
        with open(api_spec.name, "r") as f:
            api_schema = f.read()
            data_generator.generate(
                api_schema, output=output_file, output_model_type=data_generator.DataModelType.PydanticV2BaseModel
            )
        svr_tpl = jinja_env.get_template(f"api_server/{server_type}/inference.jinja.py")
        responders = { r: v['path'] for r, v in config["responders"].items() }
        if 'infer' not in responders:
            raise Exception("Server configuration must contain infer endpoint")
        req_cls = [k for k in config["responders"]["infer"]["requests"]]
        res_cls = [k for k in config["responders"]["infer"]["responses"]]
        output = svr_tpl.render(
            service_name=model_name,
            request_class=req_cls[0],
            response_class=res_cls[0],
            streaming_response_class=res_cls[0] if len(res_cls) < 2 else res_cls[1],
            endpoints=responders
        )
        with open(output_dir/"inference.py", 'w') as f:
            f.write(output)




def generate_configuration(config, tree):
    def encode_templates(templates):
        encoded_templates = dict()
        for key, value in templates.items():
            # these are json templates and need be encoded before being embeded as yaml strings
            if isinstance(value, str):
                encoded_templates[key] = base64.b64encode(value.encode())
            else:
                encoded_templates[key] = value
        return encoded_templates
    # base64 encode the templates if found in the config
    config_map = OmegaConf.to_container(config)
    try:
        for responder in config_map["server"]["responders"].values():
            input_templates = responder.get("requests", None)
            output_templates = responder.get("responses", None)
            if input_templates:
                responder["requests"] = encode_templates(input_templates)
            if output_templates:
                responder["responses"] = encode_templates(output_templates)
    except ConfigKeyError:
        raise ValueError("Server config error: responders not found")
    # write the config to a python file
    tpl_dir = get_resource_path("templates")
    jinja_env = Environment(loader=FileSystemLoader(tpl_dir))
    config_tpl = jinja_env.get_template('common/config.jinja.py')
    config = OmegaConf.create(config_map)
    output = config_tpl.render(config=OmegaConf.to_yaml(config))
    with open(tree/"config/__init__.py", 'w') as f:
        f.write(output)


def main(args):
    config = OmegaConf.load(args.config)
    with tempfile.TemporaryDirectory() as temp_dir:
        tree = build_tree(args.server_type, config, temp_dir)
        build_server(args.server_type, config.name, args.api_spec, OmegaConf.to_container(config.server), tree)
        build_inference(args.server_type, config, tree)
        generate_configuration(config, tree)
        if not args.exclude_lib :
            copy_files(get_resource_path("lib"), tree/"lib")
        if args.custom_module:
            build_custom_modules(args.custom_module, tree)
        if args.tar_output:
            target = Path(args.output_dir).resolve() / f"{config.name}.tgz"
            create_tar_gz(target, tree)
        else:
            try:
                target = Path(args.output_dir).resolve() / config.name
                shutil.copytree(tree, target, dirs_exist_ok=True)
            except FileExistsError:
                logging.error(f"{target} already exists in the output directory")
        # Build validator if validation dir provided
        if args.validation_dir:
            try:
                validation_dir = Path(args.validation_dir).resolve()
                api_spec_path = Path(args.api_spec.name).resolve()
                if validate.build_validation(api_spec_path, validation_dir, not args.no_docker):
                    logger.info("✓ Successfully built validation")
                else:
                    logger.error("✗ Failed to build validation")
            except Exception as e:
                logger.error(f"validation build failed: {e}")
                # Continue with other builds
                # Continue to finish the build without validation



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference Builder")
    build_args(parser)
    main(parser.parse_args())