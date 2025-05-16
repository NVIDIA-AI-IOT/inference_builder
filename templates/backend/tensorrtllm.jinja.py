from typing import List
from lib.inference import ModelBackend
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner, Session, TensorInfo
import tensorrt as trt
import numpy
import os

def trt_dtype_to_torch(dtype):
    '''
    Convert TRT data type to PyTorch data type
    '''
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError("%s is not supported" % dtype)

def get_trt_dtype(dtype):
    if dtype == torch.float16:
        return trt.float16
    elif dtype == torch.float32:
        return trt.float32
    elif dtype == torch.int32:
        return trt.int32
    else:
        raise TypeError("%s is not supported" % dtype)

class TensorRTLLMBackend(ModelBackend):
    """Python TensorRT Backend"""
    def __init__(self, model_config:Dict, model_home: str, device_id: int=0):
        super().__init__(model_config, model_home, device_id)
        if os.environ.get("DEBUG"):
            tensorrt_llm.logger.set_level("debug")
        self._model_name = model_config["name"]
        self._output_names = [o['name'] for o in model_config['output']]
        self._input_dtype  = { i['name']: torch_datatype_mapping[i['data_type']] for i in model_config['input']}
        self._device = f"cuda:{device_id}"
        self._stream = torch.cuda.Stream(self._device)
        torch.cuda.set_stream(self._stream)
        logger.debug(f"TensorRTBackend created for {self._model_name} to generate {self._output_names}")
        if "tensorrt_engine" not in model_config:
            raise("PolygraphBackend requires a path to tensorrt_engine")
        engine_file = model_config["parameters"]["tensorrt_engine"]
        if not os.path.isabs(engine_file):
            engine_file = os.path.join(self._model_home, engine_file)
        logger.info(f"Loading TensorRT Engine from {engine_file}...")
        with open(engine_file, 'rb') as f:
            engine_buffer = f.read()
            self._trt_session = Session.from_serialized_engine(engine_buffer)
        logger.info("TensorRT Engine loaded")

    def __call__(self, *args, **kwargs):
        # TensorRT LLM uses implicit batching, so we need to stack the input tensors
        in_data = stack_tensors_in_dict(args) if args else dict(kwargs)
        tensor_infos = []
        for key in in_data:
            tensor = in_data[key]
            if isinstance(tensor, numpy.ndarray):
                tensor = torch.from_numpy(tensor).to(self._device)
            if not isinstance(tensor, torch.Tensor):
                logger.error(f"Input tensor must be a numpy array or a torch tensor, but got {type(tensor)}")
                return
            dtype = self._input_dtype[key]
            in_data[key] = tensor.to(dtype=dtype)
            tensor_infos.append(TensorInfo(key, get_trt_dtype(dtype), tensor.shape))
        output_info = self._trt_session.infer_shapes(tensor_infos)
        trt_out = { t.name: torch.empty(tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device=tensor.device) for t in output_info }
        ok = self._trt_session.run(in_data, trt_out, self._stream.cuda_stream)
        assert ok, "Runtime execution failed for vision encoder session"
        self._stream.synchronize()
        # TODO associate trt output names with triton output names
        yield trt_out

