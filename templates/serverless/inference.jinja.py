import argparse
from config import global_config
from typing import List, Optional
import asyncio
from lib.inference import py_datatype_mapping
from .model import GenericInference
import json
from lib.utils import NumpyFlatEncoder


def create_parser(inputs: List) -> argparse.ArgumentParser:
    """Create and configure the argument parser for the application."""
    parser = argparse.ArgumentParser(
        description='Command line interface for{{ service_name }}'
    )
    for input_item in inputs:
        input_name = input_item.name.replace("_", "-")
        nargs = 1
        optional = input_item.optional if hasattr(input_item, "optional") else False
        if len(input_item.dims) == 1:
            if input_item.dims[0] == -1:
                nargs = "+"
            elif input_item.dims[0] == 1 and optional:
                nargs = "?"
            else:
                nargs = input_item.dims[0]
        parser.add_argument(
            f'--{input_name}',
            type=py_datatype_mapping[input_item.data_type],
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
    if save_to:
        delattr(args, 'save_to')
        # Create empty file if save_to is specified
        open(save_to, 'w', encoding='utf-8').close()

    # Convert remaining args to dictionary
    inputs = vars(args)

    # Convert input names back to original format (replace '-' with '_')
    inputs = {k.replace('-', '_'): v for k, v in inputs.items()}

    service = GenericInference()
    service.initialize()
    async for result in service.execute(inputs):
        if save_to:
            with open(save_to, "a", encoding='utf-8') as f:
                json_str = json.dumps(result, indent=4, cls=NumpyFlatEncoder)
                f.write(json_str)
                f.write("\n")
        else:
            print(result)
            print()

    print("Inference completed.")
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
