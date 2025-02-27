import json
import os
import sys
import argparse
import subprocess
from typing import List, Dict, Union, Any
import numpy as np
import base64
import importlib

def generate_openapi_client(output_dir_name: str, openapi_spec_path: str) -> bool:
    """Generate OpenAPI client using Docker.
    
    Args:
        output_dir_name: Name of the output directory (will be created under openapi spec's parent dir)
        openapi_spec_path: Path to OpenAPI specification file (yaml or json)
        
    Returns:
        bool: True if generation successful, False otherwise
        
    Examples:
        # Generate client with absolute paths
        generate_openapi_client(
            output_dir_name="generated_client",
            openapi_spec_path="/path/to/openapi.yaml"
        )
        # Will create /path/to/generated_client/ next to openapi.yaml
    """
    try:
        # Get absolute path of spec file
        abs_spec_path = os.path.abspath(openapi_spec_path)
        spec_dir = os.path.dirname(abs_spec_path)
        spec_file = os.path.basename(abs_spec_path)
        
        # Construct docker command
        # docker run --rm -v "${PWD}:/local" openapitools/openapi-generator-cli generate \
        #     -i /local/openapi.yaml \
        #     -g python \
        #     -o /local/generated_client
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{spec_dir}:/local",
            "openapitools/openapi-generator-cli", "generate",
            "-i", f"/local/{spec_file}",
            "-g", "python",
            "-o", f"/local/{output_dir_name}"
        ]

        print(f"Generating OpenAPI client using Docker...")
        print(f"Using OpenAPI spec: {abs_spec_path}")
        print(f"Output directory: {spec_dir}/{output_dir_name}")

        result = subprocess.run(docker_cmd, 
                              check=True,
                              capture_output=True,
                              text=True)

        if result.stdout:
            print(result.stdout)

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

def setup_environment():
    """Setup environment and generate OpenAPI client if needed."""
    generated_client_dir_name = "generated_client"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    generated_client_path = os.path.join(current_dir, generated_client_dir_name)
    
    # First check if client directory exists
    if os.path.exists(generated_client_path):
        print("✓ Found existing generated_client directory")
        # Just add to PYTHONPATH if needed
        if generated_client_path not in sys.path:
            sys.path.append(generated_client_path)
            print(f"✓ Added to PYTHONPATH: {generated_client_path}")
            return True
        else:
            print(f"✓ Already in PYTHONPATH: {generated_client_path}")
            return True

    # If directory doesn't exist, generate client
    try:
        # TODO: assuming running this script in the same dir as openapi.yaml
        openapi_spec_path = os.path.join(current_dir, "openapi.yaml")
        if not generate_openapi_client(generated_client_dir_name, openapi_spec_path):
            raise RuntimeError("Failed to generate OpenAPI client")

        # Add to PYTHONPATH
        if generated_client_path not in sys.path:
            sys.path.append(generated_client_path)
            print(f"✓ Added to PYTHONPATH: {generated_client_path}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to setup environment: {e}")
        return False

# Try to import the client
if not check_client_exists():
    if not setup_environment():
        print("✗ Failed to setup OpenAPI client")
        sys.exit(1)
    else:
        print("✓ Successfully setup OpenAPI client")

# Import the client (will succeed now)
from openapi_client.api.nvidiametropolisinferenceapi_api import NVIDIAMETROPOLISINFERENCEAPIApi
from openapi_client.api_client import ApiClient
from openapi_client.configuration import Configuration
from openapi_client.models.inference_request import InferenceRequest
from openapi_client.models.input import Input
from openapi_client.rest import ApiException

class Validator:
    """Class for validating inference endpoint implementation.
    
    Examples:
        # Basic usage
        validator = Validator()
        validator.inference(
            image_path='/path/to/image.jpg',
            expected_output_path='/path/to/expected.json'
        )
        
        # Custom host and tolerance
        validator = Validator(host="http://localhost:8000", tolerance=1e-4)
        validator.inference(
            image_path='/path/to/image.jpg',
            expected_output_path='/path/to/expected.json'
        )
    """
    
    def __init__(self, host: str = "http://127.0.0.1:8800", tolerance: float = 1e-5):
        """Initialize the Validator.
        
        Args:
            host: Host URL for the inference API
            tolerance: Default tolerance for float comparisons
        """
        self.tolerance = tolerance
        config = Configuration(host=host)
        api_client = ApiClient(config)
        self.api = NVIDIAMETROPOLISINFERENCEAPIApi(api_client)

    @staticmethod
    def is_float_equal(a: float, b: float, tolerance: float = 1e-5) -> bool:
        """Compare floats with tolerance"""
        return abs(a - b) <= tolerance

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


    @classmethod
    def load_expected_output(cls, expected_output_path: str) -> Dict:
        """Load and parse expected output JSON file.
        
        Examples:
            expected = tester.load_expected_output("path/to/expected.json")
            # expected = {
            #     "model_name": "detector_v1",
            #     "data": [{
            #         "shape": [1, 2, 3],
            #         "bboxes": [[10, 20], [30, 40]]
            #     }]
            # }
        
        Args:
            expected_output_path: Path to JSON file containing expected results
            
        Returns:
            Dict: Parsed JSON content
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        with open(expected_output_path) as f:
            return json.load(f)

    @classmethod
    def prepare_image_input(cls, image_path: str) -> str:
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


    def health_check(self):
        try:
            response = self.api.health_ready_v1_health_ready_get_with_http_info()
            assert response.status_code == 200, "Health check failed"
            assert response.data and response.data["status"] == "ready", "Server up but service not ready"
            print("✓ Health check passed")
        except Exception as e:
            print(f"✗ Health check failed: {str(e)}")
            return

    def inference(self, image_path: str, expected_output_path: str):
        """Test health check and inference endpoints"""
        try:
            # Prepare image input
            image_input = self.prepare_image_input(image_path)

            # TODO: should directly instantiate using request class
            request = {
                "input": [image_input],
                "model": "nvidia/nvdino-v2"
            }
            request = InferenceRequest.from_json(json.dumps(request))
            response = self.api.inference(request)

            # Load expected output
            with open(expected_output_path) as f:
                expected_output = json.load(f)

            # Compare responses
            actual_output = response.to_dict()
            assert self.compare_responses(actual_output, expected_output, self.tolerance), \
                "Inference response doesn't match expected output"
            
            print("✓ Inference passed")

        except Exception as e:
            print(f"✗ Inference failed: {str(e)}")
            return

def main():
    parser = argparse.ArgumentParser(description='Test NIM')
    parser.add_argument('--image', 
                       default='/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.jpg',
                       help='Path to input image')
    parser.add_argument('--expected',
                       default='./expected.sample_720p.json',
                       help='Path to expected output JSON')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                       help='Tolerance for floating point comparisons')
    parser.add_argument('--host', default="http://127.0.0.1:8800",
                       help='Host URL for the inference API')
    
    args = parser.parse_args()
    validator = Validator(host=args.host, tolerance=args.tolerance)
    validator.health_check()
    validator.inference(args.image, args.expected)

if __name__ == '__main__':
    main()
