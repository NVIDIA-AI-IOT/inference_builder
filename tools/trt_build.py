import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import onnx

def build_engine(onnx_file_path, max_batch_size=1):
    # Create a TensorRT logger and builder
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Read the ONNX file
    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed to parse the ONNX file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Set builder configurations
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB of workspace
    builder.max_batch_size = max_batch_size

    # Build and return the engine
    print("Building the engine. This might take a while...")
    engine = builder.build_engine(network, config)
    if engine:
        print("Engine built successfully!")
    return engine

def save_engine(engine, file_name):
    with open(file_name, "wb") as f:
        f.write(engine.serialize())

def main(args):
    onnx_file_path = args.input  # Your ONNX file path
    engine_file_path = args.output  # Output TensorRT engine file

    engine = build_engine(onnx_file_path)
    if engine:
        save_engine(engine, engine_file_path)
        print(f"Engine saved to {engine_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("TRT Builder")
    parser.add_argument("input", type=str, help="Onnx input file")
    parser.add_argument("-o", "--output", type=str, help="TensorRT engine file", default="model.plan")
    main(parser.parse_args())
