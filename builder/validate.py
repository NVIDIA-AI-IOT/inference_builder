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

import argparse
import json
import os
import sys
import subprocess
import base64
import importlib
from pathlib import Path
from typing import List, Dict, Union, Any
from jinja2 import Environment, FileSystemLoader
from utils import get_resource_path
import shutil
import yaml
from utils import PayloadBuilder

# Common paths - single source of truth
OUT_DIR = ".tmp"
GENERATED_CLIENT_DIR = "generated_client"
TEST_CASES_FILE = "test_cases.yaml"

# Setup paths and generate OpenAPI client
def check_client_exists() -> bool:
    """Check if OpenAPI client exists and is importable.

    Returns:
        bool: True if client exists and can be imported, False otherwise
    """
    # Check if the module can be imported
    try:
        importlib.import_module('openapi_client')
        print("Able to import OpenAPI client")
        return True
    except ImportError:
        print("Not able to import OpenAPI client")
        return False

def setup_environment(generated_client_path: Path) -> bool:
    """Setup environment for client import."""
    try:
        # First check if client directory exists
        if not generated_client_path.exists():
            print(f"Generated client directory not found at: {generated_client_path}")
            return False

        # Add to PYTHONPATH if needed
        if str(generated_client_path) not in sys.path:
            sys.path.insert(0, str(generated_client_path))  # Insert at beginning of path
            print(f"✓ Added to PYTHONPATH: {generated_client_path}")

        # Try importing to verify setup
        importlib.import_module('openapi_client')
        print("✓ Successfully imported OpenAPI client")
        return True

    except ImportError as e:
        print(f"Failed to import OpenAPI client: {e}")
        print(f"Directory contents: {os.listdir(generated_client_path)}")
        return False
    except Exception as e:
        print(f"Unexpected error setting up environment: {e}")
        return False


def check_client_valid(client_dir: Path) -> bool:
    """Check if existing OpenAPI client is valid.

    Args:
        client_dir: Path to generated client directory

    Returns:
        bool: True if client exists and is valid, False otherwise
    """
    try:
        # Check if directory exists and has key files
        if not client_dir.exists():
            return False

        required_files = [
            "setup.py",
            "openapi_client/__init__.py",
            "openapi_client/api/nvidiametropolisinferenceapi_api.py",
            "openapi_client/models/inference_request.py",
            "openapi_client/models/inference_response.py"
        ]

        for file in required_files:
            if not (client_dir / file).exists():
                print(f"Missing required file: {file}")
                return False

        # Try importing to validate
        sys_path_modified = False
        if str(client_dir) not in sys.path:
            sys.path.insert(0, str(client_dir))
            sys_path_modified = True

        try:
            import openapi_client
            from openapi_client.models.inference_request import InferenceRequest
            from openapi_client.models.inference_response import InferenceResponse
            print(f"✓ Found valid existing client at: {client_dir}")
            return True
        except ImportError as e:
            print(f"Client validation failed: {e}")
            return False
        finally:
            if sys_path_modified:
                sys.path.remove(str(client_dir))

    except Exception as e:
        print(f"Error checking client: {e}")
        return False

def generate_openapi_client(openapi_spec_path: Path, output_dir: Path, use_docker: bool = True) -> bool:
    """Generate OpenAPI client using Docker or local OpenAPI Generator.

    Args:
        openapi_spec_path: Path to OpenAPI specification file
        output_dir: Target output directory for generated client
        use_docker: Whether to use Docker for generation (default: True)
    """
    try:
        # Check if valid client already exists
        if check_client_valid(output_dir):
            print("Using existing OpenAPI client")
            return True

        print("Generating new OpenAPI client...")
        # Get absolute paths
        abs_spec_path = openapi_spec_path.resolve()
        abs_output_dir = output_dir.resolve()

        tmp_client = "tmp_generated_client"
        tmp_client_path = abs_spec_path.parent / tmp_client

        if use_docker:
            # Construct docker command
            # docker run --rm -v "${PWD}:/local" openapitools/openapi-generator-cli generate \
            #     -i /local/openapi.yaml \
            #     -g python \
            #     -o /local/generated_client
            generator_cmd = [
                "docker", "run", "--rm",
                "-v", f"{abs_spec_path.parent}:/local",
                "--user", f"{os.getuid()}:{os.getgid()}",  # Use current user's UID/GID
                "openapitools/openapi-generator-cli", "generate",
                "-i", f"/local/{abs_spec_path.name}",
                "-g", "python",
                "-o", f"/local/{tmp_client}"
            ]
            print(f"Generating OpenAPI client using Docker...")
        else:
            # Use local OpenAPI Generator
            generator_cmd = [
                "openapi-generator-cli", "generate",
                "-i", str(abs_spec_path),
                "-g", "python",
                "-o", str(tmp_client_path)
            ]
            print(f"Generating OpenAPI client using local OpenAPI Generator...")
        print(f"Using OpenAPI spec: {abs_spec_path}")
        print(f"Temporary directory: {tmp_client_path}")
        print(f"Final output directory: {output_dir.resolve()}")

        result = subprocess.run(generator_cmd,
                              check=True,
                              capture_output=True,
                              text=True)

        if result.stdout:
            print(result.stdout)

        # Move generated files to target directory
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Use shutil.copytree to copy the contents
            shutil.copytree(tmp_client_path, output_dir, dirs_exist_ok=True)
            # Clean up temp directory
            shutil.rmtree(tmp_client_path)
            return True
        except Exception as e:
            print(f"Failed to move generated client: {e}")
            return False

        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Docker command failed: {e}")
        if e.stdout:
            print("stdout:", e.stdout)
        if e.stderr:
            print("stderr:", e.stderr)
        return False
    except Exception as e:
        print(f"✗ Failed to generate client: {e}")
        return False
    finally:
        # Ensure temp directory is cleaned up
        if 'temp_client_dir' in locals() and tmp_client_path.exists():
            shutil.rmtree(tmp_client_path)




def prepare_image_input(image_path: Path) -> str:
    """Prepare image for inference by converting to base64 format.

    Examples:
        image_input = tester.prepare_image_input("path/to/image.jpg")
        # image_input = "data:image/jpg;base64,/9j/4AAQSkZJRg..."

    Args:
        image_path: Path to image file (jpg, jpeg, or png)

    Returns:
        str: Base64 encoded image with data URI scheme

    Raises:
        ValueError: If file doesn't exist or has unsupported format
        IOError: If file can't be read
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")

    ext = os.path.splitext(image_path)[1][1:].lower()
    if ext not in ['jpg', 'jpeg', 'png']:
        raise ValueError(f"Unsupported image format: {ext}. Use jpg, jpeg or png")

    with open(image_path, "rb") as f:
        image_data = f.read()
    b64_image = base64.b64encode(image_data).decode()
    return f"data:image/{ext};base64,{b64_image}"


def get_test_cases(validation_dir: Path) -> List[Dict[str, str]]:
    """Discover test cases by scanning validation directory for images.

    Args:
        validation_dir: Path to validation directory

    Returns:
        List of test cases with image and expected response paths
    """
    test_cases = []
    supported_formats = {'.jpg', '.jpeg', '.png'}

    # Find all image files
    for idx, file_path in enumerate(validation_dir.iterdir()):
        if file_path.suffix.lower() in supported_formats:
            image_name = file_path.stem
            expected_path = validation_dir / f"expected.{image_name}.json"
            text_path = validation_dir / f"{image_name}.txt"

            if expected_path.exists():
                test_cases.append({
                    "name": image_name + f"_{idx}",
                    "input": file_path.name,
                    "text": text_path.name if text_path.exists() else None,
                    "expected": expected_path.name,
                })
            else:
                print(f"Warning: Found image {file_path.name} but missing expected response: {expected_path.name}")

    print(f"Discovered {len(test_cases)} test cases:")
    for test in test_cases:
        print(f"  • {test['name']}: {test['input']} → {test['expected']}")

    return test_cases


def get_request_template(client_dir: Path) -> Dict:
    """Generate request template from OpenAPI client models.

    Args:
        client_dir: Path to generated OpenAPI client directory

    Returns:
        Dict containing request template
    """
    try:
        # Temporarily add client to path
        if str(client_dir) not in sys.path:
            sys.path.insert(0, str(client_dir))

        # Import the request model
        from openapi_client.models.inference_request import InferenceRequest
        from openapi_client.models.inference_request_input_inner import InferenceRequestInputInner

        # Create request using the model class
        input_data = [InferenceRequestInputInner("<image_placeholder>")]  # Create Input instance first
        request = InferenceRequest(
            model='nvidia/nvdino-v2',  # First allowed value from model_validate_enum
            input=input_data
        )

        # Convert to dict for JSON serialization
        template = request.to_dict()
        print("Generated request template from OpenAPI client models")
        return template

    except ImportError as e:
        print(f"Failed to import OpenAPI client models: {e}")
        raise
    except Exception as e:
        print(f"Failed to generate request template: {e}")
        raise
    finally:
        # Clean up sys.path
        if str(client_dir) in sys.path:
            sys.path.remove(str(client_dir))


def build_requests(validation_dir: Path, out_dir: Path, client_dir: Path, test_cases_abs_path: bool = False) -> bool:
    """Build request payloads from template and config.

    Args:
        validation_dir: Directory containing validation data
        out_dir: Directory to write generated files
        client_dir: Directory containing OpenAPI client
        test_cases_abs_path: Whether to use absolute paths in test_cases.yaml (default: False)
    """
    try:
        print("\nBuilding request payloads...")
        # Get request template from OpenAPI client models
        template = get_request_template(client_dir)

        # Get test cases from directory scan
        test_cases = get_test_cases(validation_dir)
        if not test_cases:
            raise ValueError("No test cases found in validation directory")

        # Store request-response mapping
        request_response_map = []

        # Generate request for each test case
        for test in test_cases:
            request = template.copy()
            image_path = validation_dir / test["input"]
            request["input"] = PayloadBuilder.prepare_image_inputs([image_path])
            if test["text"]:
                request["text"] = PayloadBuilder.prepare_text_input_from_file(validation_dir / test["text"])

            # write request file
            request_path = out_dir / f"request.{test['name']}.json"
            with open(request_path, "w") as f:
                json.dump(request, f, indent=2)
            print(f"✓ Generated request for test '{test['name']}': {request_path}")

            expected_path = validation_dir / test["expected"]

            # Add to mapping with either absolute or relative paths
            if test_cases_abs_path:
                request_path_str = str(request_path.resolve())
                expected_path_str = str(expected_path.resolve())
            else:
                # Make paths relative to validation directory
                request_path_str = os.path.relpath(request_path, out_dir)
                expected_path_str = os.path.relpath(expected_path, out_dir)

            # Add to mapping
            request_response_map.append({
                "name": test["name"],
                "request": request_path_str,
                "expected": expected_path_str
            })

        # Write request-response mapping
        test_cases_file_path = out_dir / TEST_CASES_FILE
        with open(test_cases_file_path, "w") as f:
            yaml.safe_dump(request_response_map, f, default_flow_style=False)
        print(f"✓ Generated test cases file: {test_cases_file_path}")
        print("✓ Successfully built all request payloads")
        return True
    except Exception as e:
        print(f"✗ Failed to build requests: {e}")
        return False

def build_test_runner(out_dir: Path) -> bool:
    """Generate test runner from template."""
    try:
        print("\nBuilding test runner...")

        tpl_dir = get_resource_path("templates")
        print(f"Using template directory: {tpl_dir}")

        jinja_env = Environment(loader=FileSystemLoader(tpl_dir))
        test_tpl = jinja_env.get_template("client/test_runner.jinja.py")
        output_path = out_dir / "test_runner.py"
        with open(output_path, 'w') as f:
            validator_dir = Path(__file__).parent.resolve()
            f.write(test_tpl.render(validator_dir=validator_dir))
        print(f"✓ Generated test runner: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to build test runner: {e}")
        return False

def copy_test_runner_dep_tree(out_dir: Path) -> bool:
    """Copy test runner dependency tree."""
    try:
        print("\nCopying test runner dependency tree...")
        # Copy all local dependencies
        builder_dir = Path(__file__).parent
        files_to_copy = {
            builder_dir / "validate.py": out_dir / "validate.py",
            builder_dir / "utils.py": out_dir / "utils.py",
            # Add any other dependencies here
        }

        for src, dest in files_to_copy.items():
            shutil.copy2(src, dest)
            print(f"✓ Copied {src.name} to: {dest}")
        return True
    except Exception as e:
        print(f"✗ Failed to copy test runner dependency tree: {e}")
        return False

def build_validation(openapi_spec_path: Path, validation_dir: Path, use_docker: bool = True, test_cases_abs_path: bool = False) -> bool:
    """Generate validation components."""
    try:
        print("\n=== Building Validator ===")

        # Create temp directory
        out_dir = validation_dir / OUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created temporary directory: {out_dir}")

        # 1. Generate OpenAPI client
        print("\n1. Generating OpenAPI client...")
        client_dir = out_dir / GENERATED_CLIENT_DIR
        if not generate_openapi_client(openapi_spec_path, client_dir, use_docker):
            raise Exception("Failed to generate OpenAPI client")

        # 2. Build request payloads
        print("\n2. Building request payloads...")
        if not build_requests(validation_dir, out_dir, client_dir, test_cases_abs_path):
            raise Exception("Failed to build requests")

        # 3. Generate test runner
        print("\n3. Generating test runner...")
        if not build_test_runner(out_dir):
            raise Exception("Failed to build test runner")

        # 4. Copy test runner deps like validate.py to out_dir (.tmp)
        print("\n4. Copying validate.py to out_dir...")
        if not copy_test_runner_dep_tree(out_dir):
            raise Exception("Failed to copy test runner dependency tree")

        print("\n✓ Successfully built all validation components")
        return True
    except Exception as e:
        print(f"\n✗ Failed to build validation: {e}")
        return False


class CvValidator:
    """Class for validating CV inference results"""

    @staticmethod
    def is_float_equal(a: float, b: float, tolerance: float = 1e-5) -> bool:
        """Compare floats with relative and absolute tolerance, similar to numpy.isclose

        Uses both relative and absolute tolerance for robust comparison:
        - Relative tolerance scales with the magnitude of the values
        - Absolute tolerance handles cases where values are close to zero
        - With large tolerance values, most comparisons will pass

        Formula: abs(a - b) <= (atol + rtol * max(abs(a), abs(b)))

        Args:
            a: First float value
            b: Second float value
            tolerance: Relative tolerance parameter (default: 1e-5).
                      Also used as absolute tolerance scaled down by 1000x

        Returns:
            bool: True if values are close within tolerance

        Examples:
            # Basic usage
            is_float_equal(1.0, 1.00001, 1e-4)  # True

            # Scales with magnitude (relative tolerance)
            is_float_equal(1000.0, 1000.01, 1e-4)  # True
            is_float_equal(0.001, 0.00100001, 1e-4)  # True

            # Handles near-zero values (absolute tolerance)
            is_float_equal(1e-10, 2e-10, 1e-5)  # True

            # Large tolerance passes most cases
            is_float_equal(1.0, 2.0, 1.0)  # True
        """
        # Use tolerance as relative tolerance (rtol)
        rtol = tolerance
        # Use much smaller absolute tolerance to handle near-zero values
        atol = tolerance * 1e-3

        return abs(a - b) <= (atol + rtol * max(abs(a), abs(b)))

    @classmethod
    def compare_lists(cls, list1: List[Any], list2: List[Any], tolerance: float) -> bool:
        """Compare two lists with tolerance for float values.
        For nested lists (list of lists):
        - Outer list order doesn't matter (will be sorted)
        - Inner list order DOES matter (will not be sorted)
        For regular lists (non-nested):
        - Order doesn't matter (will be sorted)

        Examples:
            # These are equal (outer order doesn't matter):
            list1 = [[1, 2], [3, 4]]
            list2 = [[3, 4], [1, 2]]  # True

            # These are NOT equal (inner order matters):
            list1 = [[1, 2], [3, 4]]
            list2 = [[2, 1], [3, 4]]  # False

            # These are equal (floats within tolerance):
            list1 = [[1.001, 2.002], [3.003, 4.004]]
            list2 = [[3.004, 4.003], [1.002, 2.001]]  # True

            # Regular lists (order doesn't matter):
            list1 = [1.0, 2.0, 3.0]
            list2 = [3.0, 1.0, 2.0]  # True

            # Regular lists with tolerance:
            list1 = [1.001, 2.002]
            list2 = [2.001, 1.000]  # True (within tolerance)

        Args:
            list1: First list to compare
            list2: Second list to compare
            tolerance: Float tolerance for comparing float values

        Returns:
            bool: True if lists are equal within tolerance, False otherwise
        """
        if len(list1) != len(list2):
            return False

        # If lists contain more lists/tuples, keep inner items as tuples but don't sort them
        if all(isinstance(x, (list, tuple)) for x in list1 + list2):
            tuples1 = [tuple(x) for x in list1]
            tuples2 = [tuple(x) for x in list2]
            # Sort the outer list only
            sorted1 = sorted(tuples1)
            sorted2 = sorted(tuples2)
            return all(
                all(cls.is_float_equal(a, b, tolerance) if isinstance(a, float) else a == b
                    for a, b in zip(item1, item2))
                for item1, item2 in zip(sorted1, sorted2)
            )

        # For lists of floats, compare with tolerance
        if all(isinstance(x, float) for x in list1 + list2):
            sorted1 = sorted(list1)
            sorted2 = sorted(list2)
            return all(cls.is_float_equal(a, b, tolerance) for a, b in zip(sorted1, sorted2))

        # For other types, use regular comparison
        return sorted(list1) == sorted(list2)

    @classmethod
    def compare_inference_result(cls, actual: Dict, expected: Dict, tolerance: float) -> bool:
        """Compare inference result dictionaries with special handling for different field types.

        Examples:
            # Basic comparison
            actual = {
                'shape': [1, 2, 3],
                'bboxes': [[10.001, 20.002], [30.003, 40.004]],
                'labels': ['car', 'person'],
                'scores': [0.9001, 0.8002]
            }
            expected = {
                'shape': [1, 2, 3],
                'bboxes': [[30.002, 40.003], [10.002, 20.001]],
                'labels': ['person', 'car'],
                'scores': [0.8001, 0.9002]
            }
            assert compare_inference_result(actual, expected, 1e-2) == True

            # Different shapes (should fail)
            actual = {'shape': [1, 2, 3]}
            expected = {'shape': [1, 2, 4]}
            assert compare_inference_result(actual, expected, 1e-5) == False

        Args:
            actual: Dictionary containing actual inference results
            expected: Dictionary containing expected inference results
            tolerance: Float tolerance for comparing numeric values

        Returns:
            bool: True if dictionaries match within tolerance
        """
        # Remove timestamp if present
        actual.pop('timestamp', None)
        expected.pop('timestamp', None)

        if actual.keys() != expected.keys():
            return False

        for key, actual_value in actual.items():
            expected_value = expected[key]

            # Handle 'shape' specially - order matters
            if key == 'shape':
                if actual_value != expected_value:
                    return False
                continue

            # Handle lists (bboxes, probs, labels)
            if isinstance(actual_value, list):
                if not cls.compare_lists(actual_value, expected_value, tolerance):
                    return False
                else:
                    print(f"  ✓ {key} passed")
                    continue

            # Handle other fields with direct comparison
            if actual_value != expected_value:
                return False

        return True

    @classmethod
    def compare_responses(cls, actual: Dict, expected: Dict, tolerance: float = 1e-5) -> bool:
        """Compare actual and expected inference API responses.

        Examples:
            # Basic comparison
            actual = {
                'model_name': 'detector_v1',
                'data': [
                    {
                        'shape': [1, 2, 3],
                        'bboxes': [[10.001, 20.002]],
                        'scores': [0.9001]
                    }
                ]
            }
            expected = {
                'model_name': 'detector_v1',
                'data': [
                    {
                        'shape': [1, 2, 3],
                        'bboxes': [[10.002, 20.001]],
                        'scores': [0.9002]
                    }
                ]
            }
            assert compare_responses(actual, expected, 1e-2) == True

            # Different model names (should fail)
            actual = {'model_name': 'detector_v1', 'data': []}
            expected = {'model_name': 'detector_v2', 'data': []}
            assert compare_responses(actual, expected) == False

        Args:
            actual: Dictionary containing actual API response
            expected: Dictionary containing expected API response
            tolerance: Float tolerance for comparing numeric values

        Returns:
            bool: True if responses match within tolerance
        """
        if actual.keys() != expected.keys():
            return False

        for key, value in actual.items():
            if key == 'data':
                # Compare data arrays
                if len(actual['data']) != len(expected['data']):
                    return False
                for actual_item, expected_item in zip(actual['data'], expected['data']):
                    if not cls.compare_inference_result(actual_item, expected_item, tolerance):
                        return False
            else:
                # Compare other top-level fields
                if actual[key] != expected[key]:
                    return False
        return True



def main():
    parser = argparse.ArgumentParser(description='Build validator components')
    parser.add_argument('--api-spec',
                       type=str,
                       required=True,
                       help='Path to OpenAPI specification file')
    parser.add_argument('--base-dir',
                       type=str,
                       default='.',
                       help='Base directory for validation')

    args = parser.parse_args()

    with open(args.api_spec) as f:
        api_schema = f.read()

    if build_validation(api_schema, Path(args.base_dir)):
        print("✓ Successfully built validation")
    else:
        print("✗ Failed to build validation")
        sys.exit(1)

if __name__ == '__main__':
    main()