from typing import List, Dict
from lib.inference import ModelBackend
import tensorrt_llm
from tensorrt_llm.runtime import Session, TensorInfo
import tensorrt as trt
import numpy
import os
import json
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

class TensorRTSession:
    def __init__(self, stream, device, engine_file, input_dtype):
        self._stream = stream
        self._device = device
        self._input_dtype = input_dtype
        torch.cuda.set_stream(self._stream)
        logger.info(f"Loading TensorRT Engine from {engine_file}...")
        with open(engine_file, 'rb') as f:
            engine_buffer = f.read()
            self._trt_session = Session.from_serialized_engine(engine_buffer)
        logger.info("TensorRT Engine loaded")

    def infer(self, in_data: Dict):
        tensor_infos = []
        for key in in_data:
            tensor = in_data[key]
            if isinstance(tensor, numpy.ndarray):
                tensor = torch.from_numpy(tensor).to(self._device)
            if not isinstance(tensor, torch.Tensor):
                logger.error(f"Input tensor must be a numpy array or a torch tensor, but got {type(tensor)}")
                return {}
            dtype = self._input_dtype[key]
            in_data[key] = tensor.to(dtype=dtype)
            tensor_infos.append(TensorInfo(key, get_trt_dtype(dtype), tensor.shape))
        output_info = self._trt_session.infer_shapes(tensor_infos)
        trt_out = { t.name: torch.empty(tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device=tensor.device) for t in output_info }
        ok = self._trt_session.run(in_data, trt_out, self._stream.cuda_stream)
        assert ok, "Runtime execution failed for vision encoder session"
        self._stream.synchronize()
        # TODO associate trt output names with triton output names
        return trt_out


class TensorRTLLMBackend(ModelBackend):
    """Python TensorRT Backend"""
    def __init__(self, model_config:Dict, model_home: str, device_id: int=0):
        super().__init__(model_config, model_home, device_id)
        self._model_name = model_config["name"]
        self._output_names = [o['name'] for o in model_config['output']]
        self._device = f"cuda:{device_id}"
        self._stream = torch.cuda.Stream(self._device)
        self._params = model_config["parameters"]
        self._inputs = model_config["input"]
        llm_backend = model_config["backend"].split("/")[-1]
        if llm_backend == "pytorch":
            from tensorrt_llm import SamplingParams
            from tensorrt_llm._torch import LLM
            from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
            from tensorrt_llm.llmapi import (EagleDecodingConfig, KvCacheConfig,
                                            MTPDecodingConfig)
            from tensorrt_llm.inputs import INPUT_FORMATTER_MAP
            in_names = [i['name'] for i in model_config['input']]
            if "max_tokens" not in in_names or "temperature" not in in_names or "top_p" not in in_names or "top_k" not in in_names:
                raise ValueError("max_tokens, temperature, top_p, and top_k must be provided for tensorrtllmpytorch backend")
            self._trt_session = None
            pytorch_config = PyTorchConfig(
                disable_overlap_scheduler=self._params.get("disable_overlap_scheduler", False),
                kv_cache_dtype=self._params.get("kv_cache_dtype", "auto"),
                attn_backend=self._params.get("attn_backend", "TRTLLM"),
                use_cuda_graph=self._params.get("use_cuda_graph", False),
                load_format=self._params.get("load_format", "auto"),
                print_iter_log=self._params.get("print_iter_log", False),
                torch_compile_enabled=self._params.get("use_torch_compile", False),
                torch_compile_piecewise_cuda_graph=self._params.get("use_piecewise_cuda_graph", False),
                moe_backend=self._params.get("moe_backend", "CUTLASS"),
                enable_trtllm_sampler=self._params.get("enable_trtllm_sampler", False)
            )
            kv_cache_config = KvCacheConfig(
                enable_block_reuse=not self._params.get("disable_kv_cache_reuse", False),
                free_gpu_memory_fraction=self._params.get("kv_cache_fraction", None),
            )

            spec_decode_algo = self._params.get("spec_decode_algo", None)
            if spec_decode_algo is not None:
                spec_decode_algo = spec_decode_algo.upper()

            if spec_decode_algo == 'MTP':
                spec_config = MTPDecodingConfig(
                    num_nextn_predict_layers=self._params.get("spec_decode_nextn", 1),
                    use_relaxed_acceptance_for_thinking=self._params.get("use_relaxed_acceptance_for_thinking", False),
                    relaxed_topk=self._params.get("relaxed_topk", 1),
                    relaxed_delta=self._params.get("relaxed_delta", 0.0))
            elif spec_decode_algo == "EAGLE3":
                spec_config = EagleDecodingConfig(
                    max_draft_len=self._params.get("spec_decode_nextn", 1),
                    pytorch_eagle_weights_path=self._params.get("eagle_model_dir", None))
            else:
                spec_config = None
            self._llm = LLM(
                model=self._model_home,
                max_model_len=self._params.get("max_model_len", 1024),
                max_batch_size=model_config.get("max_batch_size", 1),
                max_num_tokens=self._params.get("max_num_tokens", 1024),
                pytorch_backend_config=pytorch_config,
                kv_cache_config=kv_cache_config,
                tensor_parallel_size=self._params.get("tp_size", 1),
                pipeline_parallel_size=self._params.get("pp_size", 1),
                enable_attention_dp=self._params.get("enable_attention_dp", False),
                moe_expert_parallel_size=self._params.get("moe_ep_size", -1),
                moe_tensor_parallel_size=self._params.get("moe_tp_size", -1),
                moe_cluster_parallel_size=self._params.get("moe_cluster_size", -1),
                enable_chunked_prefill=self._params.get("enable_chunked_prefill", False),
                speculative_config=spec_config
            )
            self._sample_params_cls = SamplingParams
            self._input_formatter = INPUT_FORMATTER_MAP["qwen2_5_vl"]
            self._trtllm_input_name = next((i["name"] for i in self._inputs if i["data_type"] == "TYPE_CUSTOM_TRTLLM_INPUT"), None)
            if self._trtllm_input_name is None:
                raise ValueError("TYPE_CUSTOM_TRTLLM_INPUT must be provided for tensorrtllm/pytorch backend")

            logger.debug(f"TensorRTLLMBackend with pytorch created for {self._model_name} to generate {self._output_names}")
        else:
            self._llm = None
            if "tensorrt_engine" not in model_config:
                raise("TensorRTLLM backend requires a path to tensorrt_engine")
            engine_file = model_config["parameters"]["tensorrt_engine"]
            if not os.path.isabs(engine_file):
                engine_file = os.path.join(self._model_home, engine_file)
            input_dtype  = { i['name']: torch_datatype_mapping[i['data_type']] for i in model_config['input']}
            self._trt_session = TensorRTSession(self._stream, self._device, engine_file, input_dtype)
            logger.debug(f"TensorRTBackend created for {self._model_name} to generate {self._output_names}")

    def __call__(self, *args, **kwargs):
        if self._llm is not None:
            params = args[0] if args else kwargs
            sample_params = self._sample_params_cls(
                max_tokens=params.get("max_tokens", 1024),
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 0.95),
                top_k=params.get("top_k", 0)
            )
            inputs = []
            if args:
                for i in args:
                    if self._trtllm_input_name in i:
                        input = i[self._trtllm_input_name]
                        inputs.extend(input) if isinstance(input, list) else inputs.append(input)
                    else:
                        logger.warning(f"TensorRTLLMBackend: Input {self._trtllm_input_name} not found in {i}")
            elif kwargs:
                if self._trtllm_input_name in kwargs:
                    inputs = kwargs[self._trtllm_input_name]
                else:
                    logger.warning(f"TensorRTLLMBackend: Input {self._trtllm_input_name} not found in {kwargs}")
            else:
                logger.warning("TensorRTLLMBackend: No inputs provided")
            if not inputs:
                return
            inputs = self._input_formatter(self._model_home, inputs)
            results = self._llm.generate(inputs, sample_params)
            yield [{"outputs": r} for r in results]
        elif self._trt_session is not None:
            # TensorRT LLM uses implicit batching, so we need to stack the input tensors
            in_data = stack_tensors_in_dict(args) if args else dict(kwargs)
            yield self._trt_session.infer(in_data)
        else:
            raise Exception("TensorRTLLM backend is not correctly initialized")





