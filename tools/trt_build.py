import argparse
import tensorrt as trt

def build_engine(onnx_file_path, image_width, image_height, max_batch_size=1):
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
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # 2GB of workspace
    config.set_flag(trt.BuilderFlag.FP16)
#    builder.max_batch_size = max_batch_size
    profile = builder.create_optimization_profile()
    H, W = image_height, image_width
    opt = max(1, int(max_batch_size/2))
    for i in range(network.num_inputs):
        inputT = network.get_input(i)
        inputT.shape = [-1, 3, H, W]
        profile.set_shape(
            inputT.name,
            min=trt.Dims((1, 3, H, W)),
            opt=(opt, 3, H, W),
            max=(max_batch_size, 3, H, W)
        )
    config.add_optimization_profile(profile)

    # Build and return the engine
    print("Building the engine. This might take a while...")
    engine = builder.build_serialized_network(network, config)
    if engine:
        print("Engine built successfully!")
    return engine

def save_engine(engine, file_name):
    with open(file_name, "wb") as f:
        f.write(engine)

def main(args):
    onnx_file_path = args.input  # Your ONNX file path
    engine_file_path = args.output  # Output TensorRT engine file

    engine = build_engine(onnx_file_path, args.width, args.height, args.max_batch_size)
    if engine:
        save_engine(engine, engine_file_path)
        print(f"Engine saved to {engine_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("TRT Builder")
    parser.add_argument("input", type=str, help="Onnx input file")
    parser.add_argument("-o", "--output", type=str, help="TensorRT engine file", default="model.plan")
    parser.add_argument('--max_batch_size',
                        type=int,
                        default=1,
                        help="Maximum batch size for input images")
    parser.add_argument('--height',
                        type=int,
                        default=448,
                        help="Model input height")
    parser.add_argument('--width',
                        type=int,
                        default=448,
                        help="Model input width")
    main(parser.parse_args())
