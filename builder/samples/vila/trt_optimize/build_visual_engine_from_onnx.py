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


import argparse
import os
import shutil
import sys
from time import time

# isort: off
import torch
import tensorrt as trt
from tensorrt_llm.builder import Builder
# isort: on

# from PIL import Image
# from transformers import (AutoProcessor, Blip2ForConditionalGeneration,
#                           Blip2Processor, LlavaForConditionalGeneration,
#                           NougatProcessor, VisionEncoderDecoderModel)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',
                        type=str,
                        default=None,
                        choices=[
                            'opt-2.7b', 'opt-6.7b', 'flan-t5-xl', 'flan-t5-xxl',
                            'llava', 'vila', 'nougat'
                        ],
                        help="Model type")
    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        help="Directory where visual TRT engines are saved")
    parser.add_argument('--max_batch_size',
                        type=int,
                        default=4,
                        help="Maximum batch size for input images")
    parser.add_argument('--height',
                        type=int,
                        default=448,
                        help="Model input height")
    parser.add_argument('--width',
                        type=int,
                        default=448,
                        help="Model input width")
    return parser.parse_args()


class VisionEngineBuilder:

    def __init__(self, args):
        args.device = torch.device(
            "cuda") if torch.cuda.is_available() else "cpu"
        if args.output_dir is None:
            args.output_dir = 'visual_engines/%s' % (
                args.model_path.split('/')[-1])
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        self.args = args

    def build(self):
        args = self.args
        build_trt_engine(args.model_type, args.height, args.width,
                     args.output_dir, args.max_batch_size)

def export_visual_wrapper_onnx(visual_wrapper, image, output_dir):
    logger.log(trt.Logger.INFO, "Exporting onnx")
    os.makedirs(f'{output_dir}/onnx', exist_ok=True)
    torch.onnx.export(visual_wrapper,
                      image,
                      f'{output_dir}/onnx/visual_encoder.onnx',
                      opset_version=17,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {
                          0: 'batch'
                      }})


def build_trt_engine(model_type, img_height, img_width, output_dir,
                     max_batch_size):
    part_name = 'visual_encoder'
    onnx_file = '%s/onnx/%s.onnx' % (output_dir, part_name)
    engine_file = '%s/%s.engine' % (output_dir, part_name)
    config_file = '%s/%s' % (output_dir, "config.json")
    logger.log(trt.Logger.INFO, "Building TRT engine for %s" % part_name)

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config_wrapper = Builder().create_builder_config(precision="float16",
                                                     model_type=model_type)
    config = config_wrapper.trt_builder_config

    parser = trt.OnnxParser(network, logger)
    logger.log(trt.Logger.INFO, f'Loading onnx folder: {output_dir}/onnx')

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read(), os.path.abspath(onnx_file)):
            logger.log(trt.Logger.ERROR, "Failed parsing %s" % onnx_file)
            for error in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, parser.get_error(error))
        logger.log(trt.Logger.INFO, "Succeeded parsing %s" % onnx_file)

    # Delete onnx files since we don't need them now

    nBS = -1
    nMinBS = 1
    nOptBS = max(nMinBS, int(max_batch_size / 2))
    nMaxBS = max_batch_size

    logger.log(trt.Logger.INFO,
               f"Processed image dims {img_height}x{img_width}")
    H, W = img_height, img_width
    inputT = network.get_input(0)
    inputT.shape = [nBS, 3, H, W]
    profile.set_shape(inputT.name, [nMinBS, 3, H, W], [nOptBS, 3, H, W],
                      [nMaxBS, 3, H, W])
    config.add_optimization_profile(profile)

    t0 = time()
    engine_string = builder.build_serialized_network(network, config)
    t1 = time()
    if engine_string is None:
        raise RuntimeError("Failed building %s" % (engine_file))
    else:
        logger.log(trt.Logger.INFO,
                   "Succeeded building %s in %d s" % (engine_file, t1 - t0))
        with open(engine_file, 'wb') as f:
            f.write(engine_string)

    Builder.save_config(config_wrapper, config_file)

    logger.log(trt.Logger.INFO, f'deleting original onnx folder: {output_dir}/onnx')
    shutil.rmtree(f'{output_dir}/onnx')

if __name__ == '__main__':
    """ Usage
        $ python build_visual_engine_from_onnx.py --model_type vila --height 448 --width 448 \
        --max_batch_size 4 --output_dir /workspace/checkpoints/optimized/vila1.5-40b/fp16/1-gpu/visual_engines
    """
    logger = trt.Logger(trt.Logger.INFO)
    args = parse_arguments()
    builder = VisionEngineBuilder(args)
    builder.build()
