import argparse
from config import global_config
from typing import List, Optional
import asyncio
from lib.inference import py_datatype_mapping
from .model import GenericInference
import json


def create_parser(inputs: List) -> argparse.ArgumentParser:
    """Create and configure the argument parser for the application."""
    parser = argparse.ArgumentParser(
        description='Command line interface for{{ service_name }}'
    )
    for input in inputs:
        input_name = input.name.replace("_", "-")
        nargs = 1
        if len(input.dims) == 1:
            nargs = "+" if input.dims[0] == -1 else input.dims[0]
        optional = input.optional if hasattr(input, "optional") else False
        parser.add_argument(
            f'--{input_name}',
            type=py_datatype_mapping[input.data_type],
            nargs=nargs,
            required=not optional
        )
    parser.add_argument(
        '-s',
        '--save-to',
        type=str,
        help='Save results to a file'
    )
    return parser


async def run_inference(args) -> Optional[int]:
    """Run the inference service asynchronously.

    Args:
        args: Parsed command line arguments

    Returns:
        Optional[int]: Exit code, 0 for success
    """
    # Read and remove save_to argument before converting to dict
    save_to = getattr(args, 'save_to', None)
    if hasattr(args, 'save_to'):
        delattr(args, 'save_to')

    # Convert remaining args to dictionary
    inputs = vars(args)

    # Convert input names back to original format (replace '-' with '_')
    inputs = {k.replace('-', '_'): v for k, v in inputs.items()}

    service = GenericInference()
    service.initialize()
    results = []
    async for result in service.execute(inputs):
        results.append(result)

    # Save results if save_to was specified
    if save_to:
        with open(save_to, "w") as f:
            for result in results:
                json_str = json.dumps(result, indent=4)
                f.write(json_str)
                f.write("\n")
    else:
        for result in results:
            print(result)
            print()
    print(f"Inference completed with {len(results)} results")
    service.finalize()
    return 0


def main() -> int:
    """Main entry point for the inference service.

    Returns:
        Optional[int]: Exit code, None for success
    """
    parser = create_parser(global_config.input)
    args = parser.parse_args()

    try:
        return asyncio.run(run_inference(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
