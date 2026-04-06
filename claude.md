# CLAUDE.md

## Project

This Python tool generates vision AI pipelines.

It works by combining:

- inference backend templates (e.g. Deepstream, TensorRT, vLLM),
- reusable common libraries,
- custom user-defined processors,

into a complete executable inference pipeline.

## Build and Run

A Python virtual environment must exist in the project root directory.

Example:

python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt

## Project Structure

The project consists of three main modules:

- builder/ - A Python command-line tool that generates inference pipeline source code.
- lib/ - Shared libraries that will be included in the generated pipeline.
- templates/ - Jinja template files used to generate code for different servers and inference backends.

Additional modules:

- doc/ - Documentation for end users
- schemas/ - Schema definitions describing pipeline objects, configurations, and data structures used by the pipeline.
- mcp/ - MCP server implementation that exposes this tool and its functionality to AI agents.

## Working Flow

### Adding support for a new inference backend

To add a new inference backend, follow these steps:

1. Create a new template file under:
   templates/backend/<backend_name>.jinja.py

2. In this file, define a new backend class:
   <backend_name>Backend

3. The backend class must inherit from:
   ModelBackend

4. Implement the following methods:

   __init__(self, config)
       Initialize the backend using the provided configuration.

   __call__(self, *args, **kwargs)
       Run inference for one batch and yield results when available.

       Batch input behavior:
       - Explicit batch mode:
         Input tensors are provided through *args as a list.
       - Implicit batch mode:
         Input tensors are provided through **kwargs as named tensors.

       The method should yield inference results as soon as they are available.

5. Add a sample configuration and corresponding test configurations for the new backend under:
   builder/samples/<example_name>/

6. Update documentation and schema files.

### Adding support for a new API server

To implement a new API server:

Directory structure:
    templates/api_server/<api_server_name>/

Required templates:
    templates/api_server/<api_server_name>/route/inference.jinja2
    templates/api_server/<api_server_name>/responder.jinja.py

Implementation requirements:

1. inference.jinja2
    Defines the API route for inference requests.
    Handles request parsing and calls the inference pipeline.

2. responder.jinja.py
    Defines the Responder class.

    Responder responsibilities:
        - Convert incoming API requests into pipeline inputs.
        - Convert pipeline outputs into API responses.

### Adding a new responder

A responder implements a function that can be associated with a specific API endpoint.
Responder templates live under templates/responder/ and are Jinja templates that render
into async methods embedded in the Responder class.

To add a new responder:

1. Create a new template file:
   templates/responder/<responder_name>.jinja.py

2. The template must define a single async method using the Jinja variable `{{ name }}`:

       async def {{ name }}(self, request, ...):
           ...
           return <status_code>, <response>

   The method receives `self` (the Responder instance) and `request` (the HTTP request).
   Additional parameters depend on the endpoint (e.g. `body` for inference, `file` for uploads).
   It must return a tuple of (HTTP status code, response body).

3. Available helpers from the Responder base class:
   - self.process_request(name, body) - converts the API request into pipeline input
   - self.process_response(name, request, result) - converts pipeline output into an API response
   - self._inference.execute(data) - runs the inference pipeline (async generator)
   - self._asset_manager - manages file assets (save, list, delete)
   - self.logger - logger instance

4. Register the responder in the pipeline config YAML under server.responders:

       server:
         responders:
           <responder_name>:
             operation: <openapi_operation_id>
             requests:
               ...
             responses:
               ...

   The key must match the template filename (without .jinja.py).

Existing responder templates for reference:
- infer.jinja.py - runs inference with streaming support
- healthy_ready.jinja.py - health/readiness check
- add_file.jinja.py / del_file.jinja.py / list_files.jinja.py - file asset management
- add_live_stream.jinja.py / del_live_stream.jinja.py / list_live_streams.jinja.py - live stream management

## Coding Convention

### Documentation and Schema Consistency

Any code change that affects behavior, configuration, or APIs MUST include
corresponding updates to documentation and schemas. This is critical:

- **Schema changes**: If a new field is added to config YAML, test config, or
  any structured input, update the relevant JSON schema under `schemas/`.
- **Documentation changes**: If behavior, workflow, or configuration options
  change, update `schemas/README.md`, `claude.md`, and `skills/inference-builder/SKILL.md` as
  applicable.
- **Sample updates**: If a new feature or backend is added, add or update
  samples under `builder/samples/` with working examples.
- **Test updates**: If a new feature is testable, add test cases to the
  relevant test config JSON files.

Do not consider a code change complete until docs, schemas, and tests are
consistent with the new behavior.

### Naming

- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

### License Headers

- All source files must include the SPDX Apache 2.0 license header.
- Python files: `# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES...`
- Jinja templates: `{# SPDX-FileCopyrightText: ... #}`
- YAML files: `# SPDX-FileCopyrightText: ...`

### Formatting

- 4-space indentation
- F-strings for string formatting: `f"message: {variable}"`

### Type Hints

- Use type hints on function signatures with return type annotations:
  `def get_logger(name: Optional[str] = None) -> logging.Logger:`
- Import types from `typing` module (`Dict`, `List`, `Union`, `Optional`)

### Imports

- Group in order: standard library, third-party, local/relative
- Separate groups with blank lines

### Logging

- Use centralized logger: `from lib.utils import get_logger` then `self.logger = get_logger(__name__)`
- F-strings in log messages: `self.logger.info(f"Processing: {name}")`

### Error Handling

- Use the project's custom error hierarchy (`Error`, `EnhancedError`, `ErrorFactory.create()`)
- Catch specific exception types, not bare `except:`

### Configuration

- OmegaConf for configuration management
- YAML config keys use `snake_case`

### Tests

- Pytest framework with `test_*.py` file naming


## Test Framework

The test framework is a containerized integration test system driven by JSON config files.
It validates generated pipelines by building Docker images, running containers, and
checking results. It supports both serverless applications (run to completion) and
HTTP server applications (fastapi, triton, nim) that accept requests.

### Running Tests

The venv must be activated so subprocesses (builder/main.py) have access to dependencies:

    source .venv/bin/activate
    cd builder/tests
    python test_docker_builds.py --config-file <config.json> [options]

Or via the wrapper script:

    cd builder/tests
    ./run_tests.sh -c <config.json> [options]

### Test Configuration

Config files are JSON arrays of test cases. They can live anywhere in the project
(e.g. builder/tests/test_configs.json, builder/samples/tao/test_config.json).
All relative paths in a config are resolved relative to the config file's directory.

Schema: schemas/common/test_config.schema.json

### Test Flow

For each test case:

1. Download models if `models` is specified (NGC or Hugging Face)
2. Run code generation via builder/main.py with `build_args`
3. Build Docker image
4. Run container:
   - Serverless: runs to completion, checks exit code and error exports
   - HTTP server: runs detached with `--network=host`, probes health endpoint,
     sends `test_requests`, then stops container
5. If `expected_results` is set, count NDJSON objects in the result file and
   fail if the count does not match
6. If `auto_validation` is set, runs the validation script on the host
7. Collect logs and determine pass/fail

### Networking

All containers run with `--network=host`. The test client connects to the service at
`127.0.0.1:{HTTP_PORT}` locally, or the Docker bridge gateway IP in CI environments.

### Adding a New Test

1. Create the test application directory with app.yaml, processors.py, etc.
2. Add a test_config.json (or add entries to an existing one)
3. If using an HTTP server, provide an openapi.yaml
4. Optionally add a Dockerfile in the same directory as the config
   (falls back to builder/tests/Dockerfile)
5. Optionally add a validation/ directory with expected outputs for auto-validation
6. Validate the config: `python -c "from jsonschema import validate; import json; validate(json.load(open('test_config.json')), json.load(open('schemas/common/test_config.schema.json')))"`