import json
import os
import argparse
from typing import List, Dict, Union, Any
import numpy as np

# export PYTHONPATH=$PYTHONPATH:/home/byin/nim/nim-templates/builder/samples/tao/generated_client
from openapi_client.api.nvidiametropolisinferenceapi_api import NVIDIAMETROPOLISINFERENCEAPIApi
from openapi_client.api_client import ApiClient
from openapi_client.configuration import Configuration
from openapi_client.models.inference_request import InferenceRequest
from openapi_client.models.input import Input
from openapi_client.rest import ApiException

def is_float_equal(a: float, b: float, tolerance: float = 1e-5) -> bool:
    """Compare floats with tolerance"""
    return abs(a - b) <= tolerance

def compare_lists(list1: List[Any], list2: List[Any], tolerance: float) -> bool:
    """Compare two lists regardless of order, with tolerance for float values"""
    if len(list1) != len(list2):
        return False
    
    # If lists contain more lists/tuples, convert inner items to tuples for sorting
    if all(isinstance(x, (list, tuple)) for x in list1 + list2):
        sorted1 = sorted(tuple(x) for x in list1)
        sorted2 = sorted(tuple(x) for x in list2)
        # Compare each element with tolerance
        return all(
            all(is_float_equal(a, b, tolerance) if isinstance(a, float) else a == b 
                for a, b in zip(item1, item2))
            for item1, item2 in zip(sorted1, sorted2)
        )
    
    # If lists contain floats, compare with tolerance
    if all(isinstance(x, float) for x in list1 + list2):
        sorted1 = sorted(list1)
        sorted2 = sorted(list2)
        return all(is_float_equal(a, b, tolerance) for a, b in zip(sorted1, sorted2))
    
    # For other types, use regular comparison
    return sorted(list1) == sorted(list2)

def compare_inference_result(actual: Dict, expected: Dict, tolerance: float) -> bool:
    """Compare inference result dictionaries"""
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
            if not compare_lists(actual_value, expected_value, tolerance):
                return False
            continue
            
        # Handle other fields with direct comparison
        if actual_value != expected_value:
            return False
            
    return True

def compare_responses(actual: Dict, expected: Dict, tolerance: float = 1e-5) -> bool:
    """Compare actual and expected responses with tolerance"""
    if actual.keys() != expected.keys():
        return False
    
    for key, value in actual.items():
        if key == 'data':
            # Compare data arrays
            if len(actual['data']) != len(expected['data']):
                return False
            for actual_item, expected_item in zip(actual['data'], expected['data']):
                if not compare_inference_result(actual_item, expected_item, tolerance):
                    return False
        else:
            # Compare other top-level fields
            if actual[key] != expected[key]:
                return False
    return True

def test_inference(image_path: str, expected_output_path: str, tolerance: float = 1e-5):
    """Test health check and inference endpoints"""
    
    # Setup API client
    config = Configuration(host="http://127.0.0.1:8800")
    api_client = ApiClient(config)
    api = NVIDIAMETROPOLISINFERENCEAPIApi(api_client)

    # Test 1: Health Check
    try:
        response = api.health_ready_v1_health_ready_get_with_http_info()
        assert response.status_code == 200, "Health check failed"
        assert response.data and response.data["status"] == "ready", "Sever up but service not ready"
        print("✓ Health check passed")
    except Exception as e:
        print(f"✗ Health check failed: {str(e)}")
        return

    # Test 2: Inference
    try:
        # Load expected output
        # TODO: extract as a function
        with open(expected_output_path) as f:
            expected_output = json.load(f)
        
        # Prepare image input
        # TODO: extract as a function
        # Validate input image
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        ext = os.path.splitext(image_path)[1][1:].lower()
        if ext not in ['jpg', 'jpeg', 'png']:
            raise ValueError(f"Unsupported image format: {ext}. Use jpg, jpeg or png")
        with open(image_path, "rb") as f:
            image_data = f.read()
        import base64
        b64_image = base64.b64encode(image_data).decode()
        image_input = f"data:image/{ext};base64,{b64_image}"

        # Make inference request using the generated client
        request = {
            "input": [image_input],
            "model": "nvidia/nvdino-v2"
        }
        request = InferenceRequest.from_json(json.dumps(request))
        response = api.inference(request)

        # Convert response to dict for comparison
        actual_output = response.to_dict()

        # Compare responses with tolerance
        assert compare_responses(actual_output, expected_output, tolerance), \
            "Inference response doesn't match expected output"
        
        print("✓ Inference test passed")
        # print("\nActual Response:")
        # print(json.dumps(actual_output, indent=2))
        # print("\nExpected Response:")
        # print(json.dumps(expected_output, indent=2))

    except Exception as e:
        print(f"✗ Inference failed: {str(e)}")
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test NIM')
    parser.add_argument('--image', 
                       default='/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.jpg',
                       help='Path to input image (default: /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.jpg)')
    parser.add_argument('--expected',
                       default='./expected.sample_720p.json',
                       help='Path to expected output JSON (default: ./expected.sample_720p.json)')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                      help='Tolerance for floating point comparisons (default: 1e-5)')
    
    args = parser.parse_args()
    test_inference(args.image, args.expected, args.tolerance)
