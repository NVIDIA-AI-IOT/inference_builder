from typing import List
from lib.inference import ModelBackend
from tensorrt_llm.runtime import ModelRunner, Session, TensorInfo
import tensorrt as trt
import numpy

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

class TensorRTLLMBackend(ModelBackend):
    """Python TensorRT Backend"""
    def __init__(self, model_config:Dict, device_id: int=0):
        super().__init__(model_config, device_id)
        self._model_name = model_config["name"]
        self._output_names = [o['name'] for o in model_config['output']]
        self._device = f"cuda:{device_id}"
        self._stream = torch.cuda.Stream(self._device)
        torch.cuda.set_stream(self._stream)
        logger.debug(f"TensorRTBackend created for {self._model_name} to generate {self._output_names}")
        if "tensorrt_engine" not in model_config:
            raise("PolygraphBackend requires a path to tensorrt_engine")
        engine_file = model_config["tensorrt_engine"]
        logger.info(f"Loading TensorRT Engine from {engine_file}...")
        with open(engine_file, 'rb') as f:
            engine_buffer = f.read()
            self._trt_session = Session.from_serialized_engine(engine_buffer)
        logger.info("TensorRT Engine loaded")

    def __call__(self, *args, **kwargs):
        logger.debug(f"TensorRTBackend {self._model_name} triggerred with  {args if args else kwargs}")

        in_data_list = args if args else [kwargs]
        for item in in_data_list:
            for key in item:
                tensor = item[key]
                if isinstance(tensor, numpy.ndarray):
                    tensor = torch.from_numpy(tensor).unsqueeze(0).half().to(self._device)
                trt_in = {'input': tensor}
                output_info = self._trt_session.infer_shapes([TensorInfo('input', trt.DataType.HALF, tensor.shape)])
                trt_out = { t.name: torch.empty(tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device=tensor.device) for t in output_info }
                ok = self._trt_session.run(trt_in, trt_out, self._stream.cuda_stream)
                assert ok, "Runtime execution failed for vision encoder session"
                self._stream.synchronize()
                # TODO associate trt output names with triton output names
                yield {self._output_names[0]: trt_out['output']}