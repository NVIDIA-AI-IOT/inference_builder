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


import atexit
import base64
from concurrent.futures import ThreadPoolExecutor, Future
import os
import threading
from queue import Queue, Empty, Full
from abc import ABC, abstractmethod
from config import global_config
from .utils import get_logger, split_tensor_in_dict, FutureConsumer, QueueConsumer
from .codec import ImageDecoder
import custom
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Generator
import numpy as np
from collections import namedtuple
import json
from pyservicemaker import Pipeline, as_tensor
from pyservicemaker.utils import MediaExtractor, MediaChunk
import torch
from .asset_manager import AssetManager
import time
import asyncio
import inspect
from .errors import (
    Error,
    EnhancedError,
    ErrorFactory,
    ErrorCategory,
    ErrorSeverity,
    enable_global_error_collection,
    get_global_error_collector
)


logger = get_logger(__name__)

py_datatype_mapping = {
    "TYPE_UINT8": int,
    "TYPE_UINT16": int,
    "TYPE_UINT32": int,
    "TYPE_UINT64": int,
    "TYPE_INT8": int,
    "TYPE_INT16": int,
    "TYPE_INT32": int,
    "TYPE_INT64": int,
    "TYPE_FP16": float,
    "TYPE_FP32": float,
    "TYPE_FP64": float,
    "TYPE_STRING": str,
    "TYPE_CUSTOM_DS_IMAGE": str,
    "TYPE_CUSTOM_DS_MIME": str,
    "TYPE_CUSTOM_BINARY_URLS": str,
    "TYPE_CUSTOM_DS_SOURCE_CONFIG": str,
    "TYPE_CUSTOM_BINARY_BASE64": str,
    "TYPE_CUSTOM_VIDEO_CHUNK_ASSETS": str,
    "TYPE_CUSTOM_VIDEO_ASSETS": str,
    "TYPE_CUSTOM_IMAGE_ASSETS": str,
    "TYPE_CUSTOM_IMAGE_BASE64": str
}

np_datatype_mapping = {
    "TYPE_INVALID": None,
    "TYPE_BOOL": np.bool_,
    "TYPE_UINT8": np.ubyte,
    "TYPE_UINT16": np.uint16,
    "TYPE_UINT32": np.uint32,
    "TYPE_UINT64": np.uint64,
    "TYPE_INT8": np.byte,
    "TYPE_INT16": np.int16,
    "TYPE_INT32": np.int32,
    "TYPE_INT64": np.int64,
    "TYPE_FP16": np.float16,
    "TYPE_FP32": np.float32,
    "TYPE_FP64": np.float64,
    "TYPE_STRING": np.string_,
    "TYPE_CUSTOM_DS_IMAGE": np.ubyte,
    "TYPE_CUSTOM_DS_MIME": np.string_,
    "TYPE_CUSTOM_DS_SOURCE_CONFIG": str,
    "TYPE_BF16": None,
    "TYPE_CUSTOM_OBJECT": None
}

torch_datatype_mapping = {
    "TYPE_INVALID": None,
    "TYPE_BOOL": torch.bool,
    "TYPE_UINT8": torch.uint8,
    "TYPE_UINT16": torch.int16,
    "TYPE_UINT32": torch.int32,
    "TYPE_UINT64": torch.int64,
    "TYPE_INT8": torch.int8,
    "TYPE_INT16": torch.int16,
    "TYPE_INT32": torch.int32,
    "TYPE_INT64": torch.int64,
    "TYPE_FP16": torch.float16,
    "TYPE_FP32": torch.float32,
    "TYPE_FP64": torch.float64,
    "TYPE_STRING": None,
    "TYPE_CUSTOM_DS_IMAGE": torch.int8,
    "TYPE_CUSTOM_DS_MIME": None,
    "TYPE_CUSTOM_DS_SOURCE_CONFIG": str,
    "TYPE_BF16": None,
    "TYPE_CUSTOM_OBJECT": None
}

@dataclass
class Stop:
    reason: str

    def __bool__(self):
        return False

Path = namedtuple('Path', ['source', 'target'])
Route = namedtuple('Route', ['model', 'data'])

class DataFlow:
    """A single data flow from or to a model"""
    def __init__(
            self,
            configs: List[Dict],
            tensor_names: List[Tuple[str, str]],
            inbound: bool = False,
            outbound: bool = False,
            timeout=1.0
        ):
        self._configs = configs
        self._tensor_names = tensor_names
        self._inbound = inbound
        self._outbound = outbound
        self._timeout = timeout
        self._queue = Queue()
        self._optional = False
        self._stop_event = threading.Event()
        if self._inbound or self._outbound:
            self._optional = all([
                config["optional"] if "optional" in config else False
                for config in self._configs
            ])
        self._queue_consumer = None

    def _process_custom_data(self, tensor: np.ndarray, data_type: str) -> Union[List, np.ndarray, Queue, EnhancedError]:
        """
        Process custom data types that are not standard numpy types.

        Args:
            tensor: Input tensor data
            data_type: The custom data type string (e.g., "TYPE_CUSTOM_BINARY_BASE64")

        Returns:
            Union[List, np.ndarray, Queue, EnhancedError]:
                - List: For processed data like base64 decoded arrays
                - np.ndarray: For tensor data
                - Queue: For streaming data (in subclasses)
                - EnhancedError: If processing fails (e.g., decoder not initialized)

        Note:
            Subclasses override this method to handle specific data types.
            The return value is assigned to collected[o_name] in the put() method.
        """
        processed = tensor
        if self._inbound:
            if data_type == "TYPE_CUSTOM_BINARY_URLS":
                logger.debug(f"DataFlow _process_custom_data: {data_type}")
                processed = [input for input in tensor]
            elif data_type == "TYPE_CUSTOM_BINARY_BASE64":
                logger.debug(f"DataFlow _process_custom_data: {data_type}")
                processed = []
                for input in tensor:
                    if not isinstance(input, np.str_):
                        error = ErrorFactory.create(
                            "ERR_DF_002",
                            message=f"base64 binary must be bytes or string, got {type(input).__name__}",
                            caller=self,
                            input_data={"input_type": type(input).__name__},
                            expected_data={"valid_types": ["bytes", "np.bytes_", "str", "np.str_"]}
                        )
                        error.log(logger, as_json=False)
                        return error
                    try:
                        decoded = base64.b64decode(input)
                    except Exception as e:
                        error = ErrorFactory.create(
                            "ERR_DF_002",
                            message=f"Failed to decode base64 string: {e}",
                            caller=self,
                            input_data={"input": input},
                            expected_data={"valid_encoding": "base64"}
                        )
                        error.log(logger, as_json=False)
                        return error
                    processed.append(np.frombuffer(decoded, dtype=np.uint8))
        return processed

    def _is_collected_valid(self, collected: Dict):
        if not collected:
            logger.info(f"No data collected from the dataflow: {self.in_names}")
            return False
        # check output data integrity
        if self._outbound:
            for config in self._configs:
                name = config["name"]
                optional = "optional" in config and config["optional"]
                if name not in collected and not optional:
                    return False
        return True

    @property
    def in_names(self):
        return [i[0] for i in self._tensor_names]

    @property
    def o_names(self):
        return [i[1] for i in self._tensor_names]

    @property
    def optional(self):
        return self._optional

    def get_config(self, name: str):
        if not self._configs:
            return None
        return next((config for config in self._configs if config["name"] == name), None)

    def put(self, item: Union[Dict, Error, Stop]):
        if not item:
            if self._queue_consumer is None:
                # pass Error or Stop to the downstream
                self._queue.put(item, timeout=self._timeout)
            return
        # check input data integrity
        if self._inbound:
            for config in self._configs:
                name = config["name"]
                optional = "optional" in config and config["optional"]
                if name not in item and not optional:
                    error = ErrorFactory.create(
                        "ERR_DF_001",
                        message=f"{name} is not optional and not found from the dataflow input configs",
                        caller=self,
                        dataflow_names=self.in_names,
                        tensor_names=[name],
                        expected_data={"required_tensors": [c["name"] for c in self._configs if not c.get("optional", False)]},
                        actual_data={"provided_tensors": list(item.keys())},
                        related_config={"config": config, "all_configs": self._configs}
                    )
                    error.log(logger, as_json=False)
                    return
        # collect data and deposit it to the queue
        collected = {}
        for i_name, o_name in self._tensor_names:
            if i_name not in item:
                continue
            tensor = item[i_name]
            # handling custom data type
            if self._inbound or self._outbound:
                config = self.get_config(i_name)
                if config and not config["data_type"] in np_datatype_mapping and isinstance(tensor, np.ndarray):
                    result = self._process_custom_data(tensor, config["data_type"])
                    # Check if processing returned an error
                    if isinstance(result, EnhancedError):
                        self._queue.put(result)
                        return
                    collected[o_name] = result
                else:
                    collected[o_name] = tensor
            else:
                collected[o_name] = tensor
        # check output data integrity
        if not self._is_collected_valid(collected):
            error = ErrorFactory.create(
                "ERR_DF_007",
                message=f"Invalid data collected from the dataflow: {self.in_names}",
                caller=self,
                severity=ErrorSeverity.WARNING,
                dataflow_names=self.in_names,
                actual_data={"collected_keys": list(collected.keys()) if collected else []}
            )
            error.log(logger, as_json=False)
            self._queue.put(error)
            return

        #  deposit the collected data to the queue
        values = {}
        results_queues = {}
        for k, v in collected.items():
            if isinstance(v, Queue):
                results_queues[k] = v
            else:
                values[k] = v

        if results_queues:
            if len(results_queues) > 1:
                error = ErrorFactory.create(
                    "ERR_DF_004",
                    message=f"Multiple result queues are not supported in one dataflow yet, got {len(results_queues)} queues",
                    caller=self,
                    dataflow_names=self.in_names,
                    actual_data={"n_queues": len(results_queues), "queue_keys": list(results_queues.keys())}
                )
                error.log(logger, as_json=False)
                self._queue.put(error)
                return
            q_key, q = next(iter(results_queues.items()))
            def on_result_queue_result(item, user_data):
                if not item:
                    # end of the inference
                    self._queue.put(item, timeout=self._timeout)
                    return
                result_values = user_data[1]
                result_values[q_key] = item
                self._queue.put(result_values, timeout=self._timeout)
            if self._queue_consumer is None:
                self._queue_consumer = QueueConsumer("dataflow_queue_consumer", self._stop_event)
            collected.pop(q_key)
            self._queue_consumer.append_queue(q, on_result_queue_result, (q_key, collected))
        else:
            self._queue.put(values, timeout=self._timeout)

    def get(self):
        return self._queue.get(timeout=self._timeout)

    def stop(self):
        self._stop_event.set()
        self._queue.put(Stop("Shutdown"))

    def is_stopped(self):
        return self._stop_event.is_set()

    def parse_asset_string(self, asset: str):
        pieces = asset.split("?")
        asset = pieces[0]
        params = {}
        if len(pieces) > 1:
            query_string = pieces[1]
            query_params = query_string.split("&")
            for param in query_params:
                key, value = param.split("=")
                params[key] = value
        return asset, params

class VideoInputDataFlow(DataFlow):
    """A data flow for live stream data"""
    def __init__(self, configs: List[Dict], tensor_names: List[Tuple[str, str]], key_tensor_type: str, timeout=1.0):
        super().__init__(configs, tensor_names, True, False, timeout)
        self._video_tensor_type = key_tensor_type
        self._video_tensor_names = []
        for tensor_name in tensor_names:
            config = next((c for c in configs if c["name"] == tensor_name[0]), None)
            if config and config["data_type"] == key_tensor_type:
                self._video_tensor_names.append(tensor_name[1])

    def _process_custom_data(self, tensor: np.ndarray, data_type: str) -> Union[List, np.ndarray, EnhancedError]:
        """
        Process custom video data types.

        Returns:
            Union[List, np.ndarray, EnhancedError]:
                - List: Frames extracted from video assets
                - np.ndarray: Fallback to parent processing
                - EnhancedError: If asset not found or processing fails
        """
        if data_type == self._video_tensor_type:
            return self._process_video_assets(tensor)
        else:
            return super()._process_custom_data(tensor, data_type)

    def _process_video_assets(self, assets: np.ndarray):
        media_chunks = []
        results = []
        asset_list = []
        for a in assets:
            asset_manager = AssetManager()
            asset_id, params = self.parse_asset_string(a)
            asset = asset_manager.get_asset(asset_id)
            if asset:
                asset.lock()
                asset_list.append(asset)
                n_frames = int(params.get("frames", None))
                start = int(params.get("start", 0))
                duration = int(params.get("duration", asset.duration))
                interval = duration / n_frames if n_frames else 0
                media_chunks.append(MediaChunk(
                    asset.path,
                    start_pts=start,
                    duration=duration,
                    interval=interval
                ))
            else:
                error = ErrorFactory.create(
                    "ERR_DF_003",
                    message=f"Asset not found: {asset_id}",
                    caller=self,
                    input_data={
                        "asset_id": asset_id,
                        "asset_string": a,
                        "params": params
                    },
                    metadata={
                        "n_frames": params.get("frames"),
                        "start": params.get("start"),
                        "duration": params.get("duration")
                    }
                )
                error.log(logger, as_json=False)
                return error
        with MediaExtractor(media_chunks, n_thread=1) as media_extractor:
            qs = media_extractor()
            for q in qs:
                frames = []
                while True:
                    try:
                        frame = q.get(timeout=10.0)
                        if frame is None:
                            logger.info("Duration reached")
                            break
                        frames.append(frame)
                    except Empty:
                        logger.info("EOS on live streams")
                        break
                results.append(frames)
                # TODO WAR on pyservicemaker error
                time.sleep(0)
        for asset in asset_list:
            asset.unlock()
        return results

    def _is_collected_valid(self, collected: Dict):
        result = super()._is_collected_valid(collected)
        if not result:
            return False
        return all([n in collected for n in self._video_tensor_names])


class VideoFrameSamplingDataFlow(DataFlow):
    """A data flow for video frame sampling"""
    def __init__(self, configs: List[Dict], tensor_names: List[Tuple[str, str]], key_tensor_type: str, timeout=1.0):
        super().__init__(configs, tensor_names, True, False, timeout)
        n_codec_instances = int(os.getenv("N_CODEC_INSTANCES", 1))
        self._media_extractor = MediaExtractor(chunks=[], n_thread=n_codec_instances) # TODO: make it configurable
        self._video_tensor_type = key_tensor_type
        self._video_tensor_names = []
        for tensor_name in tensor_names:
            config = next((c for c in configs if c["name"] == tensor_name[0]), None)
            if config and config["data_type"] == key_tensor_type:
                self._video_tensor_names.append(tensor_name[1])
        self._media_extractor()
        self._frame_collector = ThreadPoolExecutor(max_workers=n_codec_instances)
        logger.info(f"VideoFrameSamplingDataFlow initialized")

    def _process_custom_data(self, tensor: np.ndarray, data_type: str) -> Union[Queue, List, np.ndarray, EnhancedError]:
        """
        Process custom video frame sampling data types.

        Returns:
            Union[Queue, List, np.ndarray, EnhancedError]:
                - Queue: Results queue for frame sampling (streaming)
                - List: Fallback to parent processing
                - np.ndarray: Fallback to parent processing
                - EnhancedError: If asset not found or processing fails
        """
        logger.debug(f"VideoFrameSamplingDataFlow._process_custom_data: {data_type}")
        if data_type == self._video_tensor_type:
            return self._do_video_frame_sampling(tensor)
        else:
            return super()._process_custom_data(tensor, data_type)

    def _collect_frames_from_queue(self, assets, qs, n_frames, n_chunks, results_queue):
        """Collect frames from a queue in a separate thread"""
        for _ in range(n_chunks):
            total_frames = [[] for _ in qs]
            for _ in range(n_frames):
                for i, q in enumerate(qs):
                    try:
                        frame = q.get(timeout=10.0)
                        if frame is None:
                            logger.info(f"Duration reached")
                            break
                        total_frames[i].append(frame)
                    except Empty:
                        logger.info(f"Decoder Queue is empty")
                        break
            try:
                results_queue.put(total_frames, timeout=10.0)
            except Full:
                error = ErrorFactory.create(
                    "ERR_DF_005",
                    caller=self,
                    severity=ErrorSeverity.WARNING,
                    metadata={
                        "n_frames": n_frames,
                        "n_chunks": n_chunks,
                        "queue_info": "Results queue is full"
                    }
                )
                error.log(logger, as_json=False)
                break
        results_queue.put(Stop("end"))
        for asset in assets:
            asset.unlock()
        return


    def _do_video_frame_sampling(self, assets: np.ndarray):
        qs = []
        asset_list = []
        n_frames = None
        n_chunks = None
        start = None
        duration = None
        interval = None
        for asset in assets:
            asset_manager = AssetManager()
            asset_id, params = self.parse_asset_string(asset)
            asset = asset_manager.get_asset(asset_id)
            if asset:
                asset_list.append(asset)
                n = int(params.get("frames", -1))
                if n_frames is None:
                    n_frames = n
                elif n_frames != n:
                    error = ErrorFactory.create(
                        "ERR_VIDEO_001",
                        message=f"frames mismatch on multiple assets: {n_frames} != {n}",
                        caller=self,
                        severity=ErrorSeverity.WARNING,
                        actual_data={"n_frames_first": n_frames, "n_frames_current": n}
                    )
                    error.log(logger, as_json=False)
                    return error
                n = int(params.get("chunks", 1))
                if n_chunks is None:
                    n_chunks = n
                elif n_chunks != n:
                    error = ErrorFactory.create(
                        "ERR_VIDEO_001",
                        message=f"chunks mismatch on multiple assets: {n_chunks} != {n}",
                        caller=self,
                        severity=ErrorSeverity.WARNING,
                        actual_data={"n_chunks_first": n_chunks, "n_chunks_current": n}
                    )
                    error.log(logger, as_json=False)
                    return error
                n = int(params.get("start", 0))
                if start is None:
                    start = n
                elif start != n:
                    error = ErrorFactory.create(
                        "ERR_VIDEO_001",
                        message=f"start mismatch on multiple assets: {start} != {n}",
                        caller=self,
                        severity=ErrorSeverity.WARNING,
                        actual_data={"start_first": start, "start_current": n}
                    )
                    error.log(logger, as_json=False)
                    return error
                n = int(params.get("duration", asset.duration))
                if duration is None:
                    duration = n
                elif duration != n:
                    error = ErrorFactory.create(
                        "ERR_VIDEO_001",
                        message=f"duration mismatch on multiple assets: {duration} != {n}",
                        caller=self,
                        severity=ErrorSeverity.WARNING,
                        actual_data={"duration_first": duration, "duration_current": n}
                    )
                    error.log(logger, as_json=False)
                    return error
            else:
                error = ErrorFactory.create(
                    "ERR_DF_003",
                    message=f"Asset not found: {asset_id}",
                    caller=self,
                    input_data={
                        "asset_id": asset_id,
                        "asset_string": asset,
                        "params": params
                    },
                    metadata={
                        "n_frames": n_frames,
                        "n_chunks": n_chunks,
                        "start": params.get("start", 0),
                        "duration": params.get("duration")
                    }
                )
                error.log(logger, as_json=False)
                return error

        results_queue = Queue()
        for asset in asset_list:
            asset.lock()
            interval = duration / (n_frames * n_chunks) if n_frames > 0  else 0
            chunk = MediaChunk(
                asset.path,
                start_pts=start,
                duration=duration if duration else -1,
                interval=interval if interval else 0
            )
            qs.append(self._media_extractor.append(chunk))
        self._frame_collector.submit(
            self._collect_frames_from_queue, asset_list, qs, n_frames, n_chunks, results_queue
        )
        return results_queue

    def _is_collected_valid(self, collected: Dict):
        result = super()._is_collected_valid(collected)
        if not result:
            return False
        return all([n in collected for n in self._video_tensor_names])

    def stop(self):
        super().stop()
        self._media_extractor.__del__()
        self._media_extractor = None
        self._frame_collector.shutdown(wait=True)


class ImageInputDataFlow(DataFlow):
    """A data flow for image data"""
    def __init__(self, configs: List[Dict], tensor_names: List[Tuple[str, str]], key_tensor_type: str, timeout=1.0):
        super().__init__(configs, tensor_names, True, False, timeout)
        self._image_decoder = ImageDecoder(["JPEG", "PNG"])
        self._image_tensor_names = []
        for tensor_name in tensor_names:
            config = next((c for c in configs if c["name"] == tensor_name[0]), None)
            if config and config["data_type"] == key_tensor_type:
                self._image_tensor_names.append(tensor_name[1])
        if not self._image_tensor_names:
            raise Exception("Image tensor name not found in the ImageInputDataFlow")

    def _process_custom_data(self, images: np.ndarray, data_type: str) -> Union[List, np.ndarray, EnhancedError]:
        """
        Process custom image data types.

        Args:
            images: Input image data (base64 or asset references)
            data_type: The custom data type string

        Returns:
            Union[List, np.ndarray, EnhancedError]:
                - List: Decoded images (list of tensors)
                - np.ndarray: Fallback to parent processing
                - EnhancedError: If decoder not initialized or processing fails

        Note:
            Returns EnhancedError (not raises) when decoder is not initialized,
            allowing the error to be propagated through the pipeline.
        """
        logger.debug(f"ImageInputDataFlow._process_custom_data: {data_type}")
        if self._image_decoder is None:
            return ErrorFactory.create(
                "ERR_IMG_003",
                caller=self,
                severity=ErrorSeverity.CRITICAL
            )
        if data_type == "TYPE_CUSTOM_IMAGE_BASE64":
            return self._process_base64_image(images)
        elif data_type == "TYPE_CUSTOM_IMAGE_ASSETS":
            return self._process_image_assets(images)
        else:
            return super()._process_custom_data(images, data_type)

    def _is_collected_valid(self, collected: Dict):
        result = super()._is_collected_valid(collected)
        if not result:
            return False
        return all([n in collected for n in self._image_tensor_names])

    def _process_base64_image(self, images: np.ndarray):
        result = []
        for image in images:
            if not isinstance(image, np.str_):
                error = ErrorFactory.create(
                    "ERR_IMG_002",
                    message=f"base64 image must be bytes or string, got {type(image).__name__}",
                    caller=self,
                    input_data={"image_type": type(image).__name__},
                    expected_data={"valid_types": ["bytes", "np.bytes_", "str", "np.str_"]}
                )
                error.log(logger, as_json=False)
                return error
            try:
                data_prefix, data_payload = image.split(",")
                mime_type = data_prefix.split(";")[0].split(":")[1]
                format = None
                if mime_type == "image/jpeg" or mime_type == "image/jpg":
                    format = "JPEG"
                elif mime_type == "image/png":
                    format = "PNG"
                else:
                    error = ErrorFactory.create(
                        "ERR_IMG_001",
                        message=f"Unsupported image format: {mime_type}",
                        caller=self,
                        input_data={"mime_type": mime_type, "data_prefix": data_prefix},
                        expected_data={"supported_formats": ["image/jpeg", "image/jpg", "image/png"]}
                    )
                    error.log(logger, as_json=False)
                    return error
                data_payload = base64.b64decode(data_payload)
            except Exception:
                error = ErrorFactory.create(
                    "ERR_DF_002",
                    message=f"Failed to decode base64 string for image data: {image}",
                    caller=self,
                    input_data={"input": input},
                    expected_data={"valid_encoding": "base64"}
                )
                error.log(logger, as_json=False)
                return error
            tensor = as_tensor(np.frombuffer(data_payload, dtype=np.uint8).copy(), format)
            result.append(self._image_decoder.decode(tensor, format))
        logger.debug(f"ImageInputDataFlow._process_base64_image generates {len(result)} tensors")
        return result

    def _process_image_assets(self, assets: np.ndarray):
        result = []
        for asset in assets:
            asset_manager = AssetManager()
            asset = asset_manager.get_asset(asset)
            if not asset:
                error = ErrorFactory.create(
                    "ERR_DF_003",
                    message=f"Asset not found: {asset}",
                    caller=self,
                    input_data={"asset_id": asset}
                )
                error.log(logger, as_json=False)
                return error
            asset.lock()
            format = None
            if asset.mime_type == "image/jpeg" or asset.mime_type == "image/jpg":
                format = "JPEG"
            elif asset.mime_type == "image/png":
                format = "PNG"
            else:
                error = ErrorFactory.create(
                    "ERR_IMG_001",
                    message=f"Unsupported image format: {asset.mime_type}",
                    caller=self,
                    input_data={"mime_type": asset.mime_type, "asset_path": asset.path},
                    expected_data={"supported_formats": ["image/jpeg", "image/jpg", "image/png"]}
                )
                error.log(logger, as_json=False)
                asset.unlock()
                return error
            with open(asset.path, "rb") as f:
                data = f.read()
                tensor = as_tensor(np.frombuffer(data, dtype=np.uint8).copy(), format)
                result.append(self._image_decoder.decode(tensor, format))
            asset.unlock()
        return result

inbound_dataflow_mapping = {
    "TYPE_CUSTOM_IMAGE_BASE64": ImageInputDataFlow,
    "TYPE_CUSTOM_IMAGE_ASSETS": ImageInputDataFlow,
    "TYPE_CUSTOM_VIDEO_ASSETS": VideoInputDataFlow,
    "TYPE_CUSTOM_VIDEO_CHUNK_ASSETS": VideoFrameSamplingDataFlow
}
class ModelBackend(ABC):
    """Interface for standardizing the model backend """
    def __init__(self, model_config: Dict, model_home: str, device_id=0):
        self._model_config = model_config
        self._model_home = model_home
        self._device_id = device_id

    @property
    def device_id(self):
        return self._device_id

    @property
    def model_home(self):
        return self._model_home

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise Exception("Not Implemented")

    def stop(self):
        logger.info(f'Backend for {self._model_config["name"]} stopped')

class Processor(ABC):
    def __init__(self, config: Dict, model_home: str):
        self._name = config['name']
        self._kind = config['kind'] if 'kind' in config else 'auto'
        self._input = config['input'] if 'input' in config else []
        self._output = config['output'] if 'output' in config else []
        self._config = { 'device_id': 0, 'model_home': model_home }
        if 'config' in config:
            self._config.update(config['config'])
        self._name = config['name']
        self._processor = None

    @property
    def name(self):
        return self._name

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def kind(self):
        return self._kind

    @property
    def config(self):
        return self._config

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class AutoProcessor(Processor):
    """AutoPrrocessor loads the preprocessor from pretrained"""
    def __init__(self, config: Dict, model_home: str):
        super().__init__(config, model_home)
        import transformers
        self._processor = transformers.AutoProcessor.from_pretrained(model_home)
        if self._processor is None:
            error = ErrorFactory.create(
                "ERR_PROC_002",
                message=f"Failed to load AutoProcessor from {model_home}",
                caller=self,
                severity=ErrorSeverity.CRITICAL,
                related_config={"model_home": model_home, "config": config}
            )
            error.log(logger, as_json=False)
            raise RuntimeError(str(error))

    def __call__(self, *args):
        # TODO value parser should be configurable
        return self._processor(*args)

class CustomProcessor(Processor):
    """CustomProcessor loads the processor from custom module"""
    def __init__(self, config: Dict, model_home: str):
        super().__init__(config, model_home)
        if not hasattr(custom, "create_instance"):
            error = ErrorFactory.create(
                "ERR_PROC_003",
                caller=self,
                severity=ErrorSeverity.CRITICAL,
                related_config={"config": config, "model_home": model_home}
            )
            error.log(logger, as_json=False)
            raise RuntimeError(str(error))
        self._processor = custom.create_instance(self.name, self.config)
        if self._processor is not None:
            logger.info(f"Custom processor {self._processor.name} created")
        else:
            error = ErrorFactory.create(
                "ERR_PROC_004",
                message=f"Failed to create processor {self.name}",
                caller=self,
                severity=ErrorSeverity.CRITICAL,
                related_config={"processor_name": self.name, "config": config}
            )
            error.log(logger, as_json=False)
            raise RuntimeError(str(error))

    def __call__(self, *args):
        ret = self._processor(*args)
        if not isinstance(ret, tuple):
            ret = ret,
        return ret

class Collector(ABC):
    """Collector is an interface for collecting data from the data flow"""
    def collect(self):
        raise NotImplementedError("Not implemented")

    def stop(self):
        raise NotImplementedError("Not implemented")

class SingleFlowCollector(Collector):
    """SingleFlowCollector is a collector for a single data flow"""
    def __init__(self, data_flow: DataFlow):
        self._data_flow = data_flow

    def collect(self):
        return self._data_flow.get()

    def stop(self):
        return self._data_flow.stop()

class AggregationFlowCollector(Collector):
    """AggregationFlowCollector is a collector aggregating multiple data flows"""
    def __init__(self, data_flows: List[DataFlow], timeout=None):
        self._data_flows = data_flows
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._queue = Queue()
        self._stop_event = threading.Event()
        self._timeout = timeout
        self._futures = self._executor.submit(self._collect)

    def collect(self):
        return self._queue.get(timeout=self._timeout)

    def _collect(self):
        logger.info(f"AggregationFlowCollector starts collecting data")
        while not self._stop_event.is_set():
            result = {}
            completed = []
            for data_flow in self._data_flows:
                # continue reading from each data flow until we get some data
                while not self._stop_event.is_set():
                    try:
                        data = data_flow.get()
                        if isinstance(data, Error) or isinstance(data, Stop):
                            completed.append(data)
                        else:
                            result.update(data)
                        break
                    except Empty:
                        continue
            if result:
                self._queue.put(result)
            if completed and all([isinstance(c, Stop) for c in completed]):
                self._queue.put(Stop("All data flows completed"))
            elif any([isinstance(c, Error) for c in completed]):
                self._queue.put(Error("One or more data flows ended in error"))

    def stop(self):
        logger.info(f"AggregationFlowCollector destructing...")
        self._stop_event.set()
        for data_flow in self._data_flows:
            data_flow.stop()
        self._executor.shutdown(wait=True)
        self._queue.put(Stop("Shutdown"))
        logger.info(f"AggregationFlowCollector destructed")

class MultiFlowCollector(Collector):
    """MultiFlowCollector is a collector for multiple data flows"""
    def __init__(self, data_flows: List[DataFlow], timeout=None):
        self._data_flows = data_flows
        self._executor = ThreadPoolExecutor(max_workers=len(data_flows))
        self._queue = Queue()
        self._stop_event = threading.Event()
        self._futures = [self._executor.submit(self._poll, i) for i in range(len(self._data_flows))]
        self._condition = threading.Condition()
        self._active_flow = -1
        self._timeout = timeout

    def collect(self):
        data = self._queue.get(timeout=self._timeout)
        return data

    def _poll(self, index: int):
        logger.info(f"Start polling data flow {self._data_flows[index].in_names}")
        data_flow = self._data_flows[index]
        n_data = 0
        while not self._stop_event.is_set():
            try:
                data = data_flow.get()
                is_stop = isinstance(data, Stop)
                if is_stop and n_data == 0:
                    # empty data flow, skip it
                    logger.info(f"Empty data flow {data_flow.in_names}, skip it")
                    continue
                elif not is_stop:
                    n_data += 1
                # try grabbing the queue
                with self._condition:
                    while self._active_flow != index and self._active_flow != -1:
                        self._condition.wait()
                    if self._active_flow == -1:
                        self._active_flow = index
                self._queue.put(data)
                with self._condition:
                    if isinstance(data, Stop):
                        logger.info(f"Data flow {data_flow.in_names} ended in: {data}")
                        self._active_flow = -1
                        n_data = 0
                        self._condition.notify_all()
            except Empty:
                continue


    def stop(self):
        logger.info(f"MultiFlowCollector destructing...")
        self._stop_event.set()
        for data_flow in self._data_flows:
            data_flow.stop()
        self._executor.shutdown(wait=True)
        self._queue.put(Stop("Shutdown"))
        logger.info(f"MultiFlowCollector destructed")


class AsyncDispatcher:
    """AsyncDispatcher is a dispatcher of synchronous data flows to asynchronous queues"""
    def __init__(self, collector: Collector, loop: asyncio.AbstractEventLoop, n_max_async_queues: int = 100):
        self._collector = collector
        self._loop = loop
        self._consumer_queues = Queue(maxsize=n_max_async_queues)
        self._stop_event = threading.Event()
        self._dispatcher_thread = threading.Thread(target=self.run, daemon=True)
        self._dispatcher_thread.start()

    def append_async_queue(self, async_queue: asyncio.Queue):
        try:
            self._consumer_queues.put_nowait(async_queue)
        except Full:
            error = ErrorFactory.create(
                "ERR_ASYNC_003",
                caller=self,
                severity=ErrorSeverity.CRITICAL,
                metadata={"queue_maxsize": self._consumer_queues.maxsize}
            )
            error.log(logger, as_json=False)
            raise RuntimeError(str(error))

    def run(self):
        if self._collector is None:
            error = ErrorFactory.create(
                "ERR_ASYNC_001",
                caller=self,
                severity=ErrorSeverity.CRITICAL
            )
            error.log(logger, as_json=False)
            raise RuntimeError(str(error))
        if self._loop is None:
            error = ErrorFactory.create(
                "ERR_ASYNC_002",
                caller=self,
                severity=ErrorSeverity.CRITICAL
            )
            error.log(logger, as_json=False)
            raise RuntimeError(str(error))
        logger.info("AsyncDispatcher started")
        while not self._stop_event.is_set():
            try:
                async_queue = self._consumer_queues.get(timeout=1.0)
            except Empty:
                continue
            except Exception as e:
                logger.exception(e)
                continue

            # For this consumer queue, keep collecting until Stop/Error is received
            while not self._stop_event.is_set():
                try:
                    data = self._collector.collect()
                except Empty:
                    continue
                except Exception as e:
                    logger.exception(e)
                    data = Error(str(e))
                try:
                    async def _async_put(q, item):
                        await q.put(item)
                    asyncio.run_coroutine_threadsafe(_async_put(async_queue, data), self._loop)
                except Exception as e:
                    logger.exception(e)
                    continue

                if isinstance(data, Stop):
                    logger.info(f"AsyncDispatcher delivered Stop message: {data}")
                    break

        logger.info("AsyncDispatcher stopped")

    def stop(self):
        self._stop_event.set()
        logger.info("AsyncDispatcher stopping")

class ModelOperator:
    """An model operator runs a single model"""
    def __init__(self, model_config:Dict, model_repo: str):
        self._model_name = model_config["name"]
        self._model_home = os.path.join(model_repo, self._model_name)
        self._in: List[DataFlow] = []
        self._out: List[DataFlow] = []
        self._running = False
        self._preprocessors = []
        self._postprocessors = []
        self._model_config = model_config
        self._backend = None
        self._stop_event = threading.Event()
        self._collector = None
        self._future_consumer = None


    @property
    def model_name(self):
        return self._model_name

    @property
    def model_config(self):
        return self._model_config.copy()

    @property
    def inputs(self):
        return self._in.copy()

    @property
    def outputs(self):
        return self._out.copy()

    def bind_input(self, configs: List[Dict], targets: List[str]=[]):
        if not targets:
            targets = [i['name'] for i in configs]
        flow = None
        tensor_names = [(i['name'], o) for i, o in zip(configs, targets)]
        tensor_types = [i['data_type'] for i in configs]
        image_tensor_type = None
        for tensor_type in tensor_types:
            if tensor_type in inbound_dataflow_mapping:
                image_tensor_type = tensor_type
                break
        if image_tensor_type is None:
            flow = DataFlow(configs, tensor_names, inbound=True)
        else:
            # customized inbound data flow
            flow = inbound_dataflow_mapping[image_tensor_type](configs, tensor_names, image_tensor_type)
        self._in.append(flow)
        logger.info(f"Data flow < {flow.in_names} -> {flow.o_names} > connected to model {self._model_name}")
        return flow

    def bind_output(self, configs: List[Dict], sources: List[str]=[]):
        if not sources:
            sources = [i['name'] for i in configs]
        tensor_names = [(i, o['name']) for i, o in zip(sources, configs)]
        flow = DataFlow(configs, tensor_names, outbound=True)
        self._out.append(flow)
        logger.info(f"model {self._model_name} connected to Data flow < {flow.in_names} -> {flow.o_names} >")
        return flow

    def import_input(self, input: DataFlow):
        self._in.append(input)

    def import_output(self, output: DataFlow):
        self._out.append(output)

    def set_backend(self, backend: ModelBackend):
        self._backend = backend

    def _pass_error_or_stop_downstream(self, data: Union[Error, Stop]):
        """
        Pass Error or Stop message to downstream dataflows.

        Args:
            data: Error or Stop object to pass downstream

        Note:
            If _future_consumer exists, the error/stop is wrapped in a Future
            and sent through the consumer. Otherwise, it's sent directly to
            output dataflows.
        """
        if self._future_consumer is None:
            # pass the error or stop message downstream
            for out in self._out:
                logger.debug(f"Passing error or stop message to {out.in_names}")
                out.put(data)
        else:
            f = Future()
            f.set_result(data)
            self._future_consumer.append_future(f)

    def run(self):
        logger.debug(f"Model operator for {self._model_name} started")

        # create preprocessors
        for kind, processors in [("preprocessors", self._preprocessors), ("postprocessors", self._postprocessors)]:
            if kind in self._model_config:
                for config in self._model_config[kind]:
                    ProcessorClass = None
                    if config["kind"] == "auto":
                        ProcessorClass = AutoProcessor
                    elif config["kind"] == "custom":
                        ProcessorClass = CustomProcessor
                    if ProcessorClass is not None:
                        processors.append(
                            ProcessorClass(
                                config=config,
                                model_home=self._model_home
                            )
                        )
                    else:
                        raise Exception("Invalid Processor")
        # backend loop
        self._collector = self._create_collector()
        if not self._out:
            error = ErrorFactory.create(
                "ERR_MO_003",
                caller=self,
                model_name=self._model_name,
                severity=ErrorSeverity.CRITICAL,
                related_config={"model_config": self._model_config}
            )
            error.log(logger, as_json=False)
            raise RuntimeError(str(error))
        while not self._stop_event.is_set():
            try:
                # collect input data until Stop is received
                data = self._collector.collect()
                if isinstance(data, Stop) or isinstance(data, Error):
                    self._pass_error_or_stop_downstream(data)
                    continue
                logger.info(f"Input collected from {self._collector.__class__.__name__}: {data}")

                # convert data to args and kwargs based on if explicit batching is required
                args = []
                kwargs = data
                values = [v for v in data.values()]
                lengths = [
                    len(v) if isinstance(v, list) or v.ndim > 0 else 0
                    for v in values
                ]
                if any(isinstance(v, list) for v in values) and \
                   all(length == lengths[0] for length in lengths):
                    logger.info(
                        "Explicit batching detected, "
                        "splitting the data into multiple inference requests"
                    )
                    # construct multiple inference requests
                    args = split_tensor_in_dict(kwargs)
                    kwargs = {}
                # call preprocess() before passing args to the backend
                processed, passthrough_tensors = self._preprocess(args if args else [kwargs])
                in_names = [i["name"] for i in self._model_config["input"]]
                if not processed:
                    error = ErrorFactory.create(
                        "ERR_MO_001",
                        caller=self,
                        model_name=self._model_name,
                        input_data={
                            "args": str(args)[:500] if args else "",
                            "kwargs_keys": list(kwargs.keys()) if kwargs else []
                        },
                        related_config={
                            "preprocessors": [p.name for p in self._preprocessors],
                            "model_input": in_names
                        }
                    )
                    error.log(logger, as_json=False)
                    self._pass_error_or_stop_downstream(error)
                    continue
                if args:
                    # explicitly batched data
                    args = processed
                elif kwargs:
                    # implicitly batched data
                    kwargs = processed[0]
                else:
                    error = ErrorFactory.create(
                        "ERR_MO_004",
                        message="Invalid result from preprocess: both args and kwargs are empty",
                        caller=self,
                        model_name=self._model_name,
                        input_data={
                            "processed": str(processed)[:500] if processed else "",
                            "passthrough_tensors_keys": list(passthrough_tensors[0].keys()) if passthrough_tensors else []
                        }
                    )
                    error.log(logger, as_json=False)
                    self._pass_error_or_stop_downstream(error)
                    continue
                # execute inference backend and collect result
                logger.debug(f"Model {self._model_name} invokes backend {self._backend.__class__.__name__} with {args if args else kwargs}")
                rs = self._backend(*args, **kwargs)
                if isinstance(rs, Future):
                    def on_future_result(result: Dict|Error|Stop, user_data: List):
                        if isinstance(result, Error) or isinstance(result, Stop):
                            # Pass error/stop directly - already in FutureConsumer callback context
                            for out in self._out:
                                out.put(result)
                            return
                        self._on_inference_result(result, user_data[0])
                    if self._future_consumer is None:
                        self._future_consumer = FutureConsumer(
                            name=self._model_name,
                            stop_event=self._stop_event,
                            result_callback=on_future_result
                        )
                    self._future_consumer.append_future(rs, passthrough_tensors)
                elif inspect.isgenerator(rs):
                    self._on_inference_result(rs, passthrough_tensors)
                elif isinstance(rs, Error) or isinstance(rs, Stop):
                    self._pass_error_or_stop_downstream(rs)
                    continue
                else:
                    error_msg = (
                        f"Invalid result from backend {self._backend.__class__.__name__}: {rs}"
                        " Expected a Future or a generator"
                    )
                    raise ValueError(error_msg)
            except Empty:
                continue
            except Exception as e:
                error = ErrorFactory.from_exception(
                    e,
                    caller=self,
                    category=ErrorCategory.BACKEND if "backend" in str(e).lower() else ErrorCategory.INTERNAL,
                    model_name=self._model_name,
                    dataflow_names=[o.in_names for o in self._out],
                    input_data={
                        "data_keys": list(data.keys()) if 'data' in locals() and isinstance(data, dict) else []
                    },
                    related_config={
                        "model_config": {
                            "name": self._model_config.get("name"),
                            "input": [i["name"] for i in self._model_config.get("input", [])],
                            "output": [o["name"] for o in self._model_config.get("output", [])]
                        },
                        "backend": self._backend.__class__.__name__ if self._backend else None,
                        "n_preprocessors": len(self._preprocessors),
                        "n_postprocessors": len(self._postprocessors)
                    }
                )
                error.log(logger, as_json=False)

                # Pass error downstream
                self._pass_error_or_stop_downstream(error)

        logger.info(f"Model operator {self._model_name} stopped")

    def stop(self):
        logger.info(f"Model operator {self._model_name} is stopping")
        self._stop_event.set()
        if self._collector is not None:
            self._collector.stop()
        self._in.clear()
        self._out.clear()
        self._preprocessors.clear()
        self._postprocessors.clear()

    def _on_inference_result(self, results: Generator, passthrough_tensors: Dict):
        for r in results:
            logger.debug(
                    "Model %s generated result from backend %s: %s",
                    self._model_name, self._backend.__class__.__name__, r
                )
            # iterate the result list and postprocess each of them
            for out in self._out:
                output_data = {n : [] for n in out.in_names}
                if isinstance(r, list):
                    # we get a batch
                    if len(passthrough_tensors) == 1:
                        passthrough_tensors = passthrough_tensors*len(r)
                    for i, result in enumerate(r):
                        if passthrough_tensors:
                            result.update(passthrough_tensors[i])
                        result = self._postprocess(result)
                        if not all([n in result for n in out.in_names]):
                            error = ErrorFactory.create(
                                "ERR_MO_002",
                                message=f"Model output incomplete (batch item {i})",
                                caller=self,
                                model_name=self._model_name,
                                dataflow_names=out.in_names,
                                expected_data={"required_outputs": out.in_names},
                                actual_data={"provided_outputs": list(result.keys())},
                                related_config={"postprocessors": [p.name for p in self._postprocessors]}
                            )
                            error.log(logger, as_json=False)
                            continue
                        # collect the result
                        for n, v in output_data.items():
                            if n in result:
                                v.append(result[n])
                            else:
                                v.append(None)
                else:
                    # implicit batching
                    if passthrough_tensors:
                        r.update(passthrough_tensors[0])
                    output_data = self._postprocess(r)
                    if not all([n in output_data for n in out.in_names]):
                        error = ErrorFactory.create(
                            "ERR_MO_002",
                            message="Model output incomplete",
                            caller=self,
                            model_name=self._model_name,
                            dataflow_names=out.in_names,
                            expected_data={"required_outputs": out.in_names},
                            actual_data={"provided_outputs": list(output_data.keys())},
                            related_config={"postprocessors": [p.name for p in self._postprocessors]}
                        )
                        error.log(logger, as_json=False)
                        continue
                logger.debug(f"ModelOperator of {self._model_name} deposits result: {output_data}")
                out.put(output_data)

    def _preprocess(self, args: List):
        # go through the preprocess chain
        outcome = args
        for preprocessor in self._preprocessors:
            result = []
            for data in outcome:
                # initialize the processed as the original values
                processed = {k: v for k, v in data.items()}
                # trigger the preprocessor if all the input tensors are present
                if all([i in data for i in preprocessor.input]):
                    input = [processed.pop(i) for i in preprocessor.input]
                    logger.debug(f"{self._model_name} invokes preprocessor {preprocessor.name} with given input {input}")
                    output = preprocessor(*input)
                    logger.debug(f"{self._model_name} preprocessor {preprocessor.name} generated output {output}")
                    if len(output) != len(preprocessor.output):
                        error = ErrorFactory.create(
                            "ERR_PROC_001",
                            message=f"Number of preprocessing output doesn't match the configuration, expecting {len(preprocessor.output)}, while getting {len(output)}",
                            caller=self,
                            severity=ErrorSeverity.WARNING,
                            model_name=self._model_name,
                            expected_data={"output_count": len(preprocessor.output), "output_names": preprocessor.output},
                            actual_data={"output_count": len(output)},
                            related_config={"preprocessor": preprocessor.name}
                        )
                        error.log(logger, as_json=False)
                        continue
                    # update as processed
                    for key, value in zip(preprocessor.output, output):
                        if value is not None:
                            processed[key] = value
                else:
                    logger.info(
                        "Pre-processor %s skipped on mismatching input",
                        preprocessor.name
                    )
                result.append(processed)
            # update outcome
            outcome = result
        # correct the data type and extract the passthrough tensors
        passthrough_tensors = []
        for data in outcome:
            passthrough_tensor = {}
            for key in data:
                value = data[key]
                i_config = next((i for i in self._model_config['input'] if i["name"] == key), None)
                if i_config is None:
                    logger.info(f"{key} from preprocessed is not found in the model input config, adding it as a passthrough tensor")
                    passthrough_tensor[key] = value
                else:
                    data_type = i_config["data_type"]
                    if isinstance(value, np.ndarray):
                        data_type = np_datatype_mapping[data_type]
                        if value.dtype != data_type:
                            data[key] = value.astype(data_type)
                    elif isinstance(value, torch.Tensor):
                        data_type = torch_datatype_mapping[data_type]
                        if value.dtype != data_type:
                            data[key] = value.to(data_type)
            for key in passthrough_tensor:
                data.pop(key)
            passthrough_tensors.append(passthrough_tensor)
        return outcome, passthrough_tensors

    def _postprocess(self, data: Dict):
        processed = {k: v for k, v in data.items()}
        for processor in self._postprocessors:
            if not all([i in data for i in processor.input]):
                logger.info(
                        "Pre-processor %s skipped on mismatching input",
                        processor.name
                    )
                continue
            input = [processed.pop(i) for i in processor.input]
            logger.debug(f"Post-processor {processor.name} invoked with given input {input}")
            output = processor(*input)
            logger.debug(f"Post-processor generated output {output}")
            if len(output) != len(processor.output):
                error = ErrorFactory.create(
                    "ERR_PROC_001",
                    message=f"Number of postprocessing output doesn't match the configuration, expecting {len(processor.output)}, while getting {len(output)}",
                    caller=self,
                    severity=ErrorSeverity.WARNING,
                    model_name=self._model_name,
                    expected_data={"output_count": len(processor.output), "output_names": processor.output},
                    actual_data={"output_count": len(output)},
                    related_config={"postprocessor": processor.name}
                )
                error.log(logger, as_json=False)
                continue
            # update as processed
            for key, value in zip(processor.output, output):
                processed[key] = value
        return processed

    def _create_collector(self):
        if len(self._in) == 1:
            logger.info(f"Single data flow input detected, using single flow collector on model {self._model_name}")
            return SingleFlowCollector(self._in[0])
        else:
            outputs = [set(d.o_names) for d in self._in]
            intersection = set.intersection(*outputs)
            if len(intersection) == 0 and not any([d.optional for d in self._in]):
                logger.info(f"Aggregation data flow input detected, using aggregation flow collector on model {self._model_name}")
                return AggregationFlowCollector(self._in, timeout=1.0)
            else:
                logger.info(f"Multi data flow input detected, using multi flow collector on model {self._model_name}")
                return MultiFlowCollector(self._in, timeout=1.0)

class InferenceBase:
    """The base model that drives the inference flow"""
    def initialize(self):
        # Enable global error collection FIRST - before any error can occur
        enable_global_error_collection(max_errors=5000)
        atexit.register(InferenceBase._export_errors)
        logger.info("Enhanced error collection enabled (with atexit export)")

        def parse_route(i1, i2) -> Route:
            m1 = ''
            m2 = ''
            d1 = []
            d2 = []
            s = i1.split(':')
            t = i2.split(':')
            if len(s) > 2 or len(t) > 2:
                error = ErrorFactory.create(
                    "ERR_ROUTE_001",
                    message=f"Invalid route format: {i1} -> {i2}",
                    component="InferenceBase",
                    operation="initialize",
                    severity=ErrorSeverity.CRITICAL,
                    input_data={"source": i1, "target": i2}
                )
                error.log(logger, as_json=False)
                raise RuntimeError(str(error))
            else:
                m1 = s[0]
                m2 = t[0]
                if len(s) == 2 and s[1]:
                    data = json.loads(s[1])
                    if isinstance(data, list):
                        d1 = data
                    else:
                        error = ErrorFactory.create(
                            "ERR_ROUTE_001",
                            message="Invalid route: source data must be a list",
                            component="InferenceBase",
                            operation="initialize.parse_route",
                            severity=ErrorSeverity.CRITICAL,
                            input_data={"source_data": data}
                        )
                        error.log(logger, as_json=False)
                        raise RuntimeError(str(error))
                if len(t) == 2 and t[1]:
                    data = json.loads(t[1])
                    if isinstance(data, list):
                        d2 = data
                    else:
                        error = ErrorFactory.create(
                            "ERR_ROUTE_001",
                            message="Invalid route: target data must be a list",
                            component="InferenceBase",
                            operation="initialize.parse_route",
                            severity=ErrorSeverity.CRITICAL,
                            input_data={"target_data": data}
                        )
                        error.log(logger, as_json=False)
                        raise RuntimeError(str(error))

            return Route(Path(m1, m2), Path(d1, d2))


        self._operators: ModelOperator = []
        self._inputs: List[DataFlow] = []
        self._outputs: List[DataFlow] = []
        self._input_config = OmegaConf.to_container(global_config.input)
        self._output_config = OmegaConf.to_container(global_config.output)
        self._model_repo = global_config.model_repo
        self._ready = False

        if not os.path.exists(self._model_repo):
            error = ErrorFactory.create(
                "ERR_SYS_001",
                message=f"Model repository {self._model_repo} does not exist",
                caller=self,
                severity=ErrorSeverity.CRITICAL,
                related_config={"model_repo": self._model_repo}
            )
            error.log(logger, as_json=False)
            raise RuntimeError(str(error))

        # set up the inference flow
        for model_config in global_config.models:
            self._operators.append(ModelOperator(OmegaConf.to_container(model_config), self._model_repo))

        # initialize the operators
        for operator in self._operators:
            model_config = next((m for m in global_config.models if m.name == operator.model_name), None)
            model_home = os.path.join(self._model_repo, operator.model_name)
            backend_spec = model_config.backend.split('/')
            backend_instance = self._create_backend(backend_spec, model_config, model_home)
            if backend_instance is None:
                raise RuntimeError(f"Unable to create backend {model_config.backend}")
            operator.set_backend(backend_instance)

        if hasattr(global_config, "routes"):
            # go through the routing table
            for k, v in global_config.routes.items():
                route = parse_route(k, v)
                logger.debug(f"Adding route {route}")
                # neither source nor target model is specified.
                if not route.model.source and not route.model.target:
                    # this is a direct passthrough in the top level, we can use a standalone dataflow
                    if not route.data.source and not route.data.target:
                        logger.error(f"Invalid route: {route}, source or target is required for a direct pass")
                        continue
                    dataflow = None
                    if route.data.source:
                        s_configs = [OmegaConf.to_container(c) for c in global_config.input if c.name in route.data.source]
                        if len(s_configs) != len(route.data.source):
                            logger.error(f"Not all the sources are found in the input configs, unable to create passthrough dataflow")
                            continue
                        o_tensor_names = route.data.target if route.data.target else route.data.source
                        if any(n not in [c.name for c in global_config.output] for n in o_tensor_names):
                            logger.warning(f"Not all the output tensors from {o_tensor_names} are found in the output configs, consider adding a postprocessor to generate the missing tensors")
                        tensor_names = [(i['name'], o) for i, o in zip(s_configs, o_tensor_names)]
                        dataflow = DataFlow(configs=s_configs, tensor_names=tensor_names, inbound=True)
                    elif route.data.target:
                        t_configs = [OmegaConf.to_container(c) for c in global_config.output if c.name in route.data.target]
                        if len(t_configs) != len(route.data.target):
                            logger.error(f"Not all the targets are found in the output configs, unable to create passthrough dataflow")
                            continue
                        if any(n not in [c.name for c in global_config.input] for n in route.data.target):
                            logger.error(f"Not all the targets are found in the input configs, unable to create passthrough dataflow")
                            continue
                        tensor_names = [(i, i) for i in route.data.target]
                        dataflow = DataFlow(configs=t_configs, tensor_names=tensor_names, outbound=True)
                    else:
                        logger.error(f"Invalid route: {route}, source or target is required for a direct pass")
                        continue
                    if dataflow is not None:
                        self._inputs.append(dataflow)
                        self._outputs.append(dataflow)
                elif route.model.target:
                    operator1 = next((o for o in self._operators if o.model_name == route.model.target), None)
                    if operator1 is None:
                        error = ErrorFactory.create(
                            "ERR_ROUTE_002",
                            message=f"Model {route.model.target} in the routes not found",
                            caller=self,
                            severity=ErrorSeverity.CRITICAL,
                            input_data={"model_name": route.model.target},
                            related_config={"available_models": [o.model_name for o in self._operators]}
                        )
                        error.log(logger, as_json=False)
                        raise RuntimeError(str(error))
                    # both source and target model are specified.
                    if route.model.source:
                        # this is the model provides data
                        operator2 = next((o for o in self._operators if o.model_name == route.model.source), None)
                        if operator2 is None:
                            error = ErrorFactory.create(
                                "ERR_ROUTE_002",
                                message=f"Model {route.model.source} in the routes not found",
                                caller=self,
                                severity=ErrorSeverity.CRITICAL,
                                input_data={"model_name": route.model.source},
                                related_config={"available_models": [o.model_name for o in self._operators]}
                            )
                            error.log(logger, as_json=False)
                            raise RuntimeError(str(error))
                        flow = None
                        if route.data.source and route.data.target:
                            tensor_names = [(i, o) for i, o in zip(route.data.source, route.data.target)]
                            flow = DataFlow(configs=None, tensor_names=tensor_names)
                        elif route.data.source:
                            tensor_names = [(i, i) for i in route.data.source]
                            flow = DataFlow(configs=None, tensor_names=tensor_names)
                        elif route.data.target:
                            tensor_names = [(i, i) for i in route.data.target]
                            flow = DataFlow(configs=None, tensor_names=tensor_names)
                        else:
                            logger.error(f"Invalid route: {route}, source or target is required for connecting two models, {operator2.model_name} and {operator1.model_name}")
                            continue
                        operator2.import_output(flow)
                        operator1.import_input(flow)
                    else:
                        # this is the top level input
                        if route.data.source:
                            s_configs = []
                            for s in route.data.source:
                                c = next((c for c in global_config.input if c.name == s), None)
                                if c is None:
                                    logger.error(f"Input {s} not found in the input configs, unable to create dataflow")
                                    continue
                                s_configs.append(OmegaConf.to_container(c))
                        else:
                            s_configs = [OmegaConf.to_container(c) for c in global_config.input]
                        self._inputs.append(operator1.bind_input(s_configs, route.data.target))
                # only source model is specified.
                elif route.model.source:
                    # this is the top level output
                    operator = next((o for o in self._operators if o.model_name == route.model.source), None)
                    if operator is None:
                        error = ErrorFactory.create(
                            "ERR_ROUTE_002",
                            message=f"Model {route.model.source} in the routes not found",
                            caller=self,
                            severity=ErrorSeverity.CRITICAL,
                            input_data={"model_name": route.model.source},
                            related_config={"available_models": [o.model_name for o in self._operators]}
                        )
                        error.log(logger, as_json=False)
                        raise RuntimeError(str(error))
                    if route.data.target:
                        configs = [OmegaConf.to_container(c) for c in global_config.output if c.name in route.data.target]
                        self._outputs.append(operator.bind_output(configs, route.data.source))
                    else:
                        configs = [OmegaConf.to_container(c) for c in global_config.output if c.name in route.data.source]
                        if route.data.source == [c['name'] for c in configs]:
                            self._outputs.append(operator.bind_output(configs))
                        else:
                            logger.warning(f"Output of {operator.model_name} is not compatible with top level output, be sure to add a top level postprocessor")
                            dataflow = DataFlow(configs=None, tensor_names=[(i, i) for i in route.data.source])
                            operator.import_output(dataflow)
                            self._outputs.append(dataflow)
                else:
                    logger.warning("Empty route entry")
        elif  len(self._operators) == 1:
            # default flow for single model use case without an explicit route
            operator = self._operators[0]
            # direct connection from top level input to the model
            self._inputs.append(operator.bind_input(OmegaConf.to_container(global_config.input)))
            # direct connection from the model to top level output
            self._outputs.append(operator.bind_output(OmegaConf.to_container(global_config.output)))
        else:
            logger.error("Unable to set up inference routes")
        if len(self._inputs) == 0 or len(self._outputs) == 0:
            error = ErrorFactory.create(
                "ERR_SYS_002",
                message="Either input or output is empty, inference pipeline is not complete",
                caller=self,
                severity=ErrorSeverity.CRITICAL,
                actual_data={
                    "n_inputs": len(self._inputs),
                    "n_outputs": len(self._outputs)
                },
                related_config={
                    "models": [o.model_name for o in self._operators]
                }
            )
            error.log(logger, as_json=False)
            raise RuntimeError(str(error))
        self._executor = ThreadPoolExecutor(max_workers=len(self._operators))
        self._future = None

        # sanity check on vision pipeline
        try:
            Pipeline("vision")
        except Exception as e:
            logger.exception(e)

        # post processing:
        self._processors = []
        if hasattr(global_config, "postprocessors"):
            configs = OmegaConf.to_container(global_config.postprocessors)
            for config in configs:
                if config["kind"] == "custom":
                    self._processors.append(
                        CustomProcessor(config, self._model_repo)
                    )

        # data collector
        self._collector = (
            AggregationFlowCollector(self._outputs)
            if len(self._outputs) > 1
            else SingleFlowCollector(self._outputs[0])
        )
        # async dispatcher
        self._async_dispatcher = None
        # stop event for request lifecycle
        self._stop_event = threading.Event()


        # start the inference flow
        for operator in self._operators:
            self._submit(operator)

        self._ready = True
        logger.info("Inference Engine initialized for %s", global_config.name)
        logger.info("Inputs: %s", [f.o_names for f in self._inputs])
        logger.info("Outputs: %s", [f.o_names for f in self._outputs])

    def is_healthy(self):
        return self._ready

    def finalize(self):
        if self._stop_event is not None:
            self._stop_event.set()
        if self._async_dispatcher is not None:
            self._async_dispatcher.stop()
            self._async_dispatcher = None
        if self._collector is not None:
            self._collector.stop()
            self._collector = None
        for operator in self._operators:
            operator.stop()
        self._executor.shutdown()

        logger.info("Inference pipeline is finalized")

    @staticmethod
    def _export_errors():
        """Export collected errors to JSON file for test validation.

        Uses ERROR_EXPORT_PATH env var to determine the full file path
        (defaults to /tmp/inference_errors.json).
        """
        error_export_path = os.environ.get("ERROR_EXPORT_PATH", "/tmp/inference_errors.json")
        try:
            collector = get_global_error_collector()
            if collector:
                stats = collector.get_stats(include_recent=100)
                if stats["total_errors"] > 0:
                    collector.export_to_json(error_export_path, include_stack_traces=False)
                    logger.info(f"Exported {stats['total_errors']} errors to {error_export_path}")
                else:
                    # Write empty stats file to indicate no errors
                    with open(error_export_path, 'w') as f:
                        json.dump({"total_errors": 0, "errors": []}, f)
                    logger.info(f"No errors to export, wrote empty file to {error_export_path}")
        except Exception as e:
            logger.warning(f"Failed to export errors: {e}")

    def _create_backend(self, backend_spec: List[str], model_config: Dict, model_home: str) -> ModelBackend | None:
        raise NotImplementedError("Subclass must implement this method")

    def _on_operator_done(self, future: Future, operator: ModelOperator):
        """Callback when an operator thread completes. Handles exceptions by logging to error JSON."""
        try:
            future.result()  # This raises if the thread had an exception
        except Exception as e:
            error = ErrorFactory.from_exception(
                e,
                caller=self,
                category=ErrorCategory.INTERNAL,
                model_name=operator.model_name,
                related_config={
                    "model_config": {
                        "name": operator.model_config.get("name"),
                        "preprocessors": operator.model_config.get("preprocessors", []),
                        "postprocessors": operator.model_config.get("postprocessors", [])
                    }
                }
            )
            error.log(logger, as_json=False)

            logger.critical(f"Operator {operator.model_name} thread failed with exception: {e}")
            logger.critical("Forcing application exit due to model thread failure")

            # Export errors and force exit - no graceful shutdown for unrecoverable critical exception
            InferenceBase._export_errors()
            os._exit(1)

    def _submit(self, op: ModelOperator):
        future = self._executor.submit(lambda: op.run())
        future.add_done_callback(lambda f: self._on_operator_done(f, op))