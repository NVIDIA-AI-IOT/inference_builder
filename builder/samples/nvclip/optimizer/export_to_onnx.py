# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pathlib import Path

import open_clip
import torch
from PIL import Image
import os

# Register the model config
open_clip.add_model_config("configs")

MODEL = os.environ.get("MODEL_NAME", "NVCLIP_224_700M_ViTH14")
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL,
    pretrained=f"/workspace/checkpoints/baseline/{os.environ.get("CHECKPOINT_NAME")}",
)
tokenizer = open_clip.get_tokenizer(MODEL)

text = tokenizer(["a diagram", "a dog", "a cat"])
image = preprocess(Image.open("CLIP.png")).unsqueeze(0)

# Export to onnx
npx = 224
dummy_image = torch.randn(10, 3, npx, npx)
model.forward(dummy_image,text) # Original CLIP result (1)

# Vision model
torch.onnx.export(model, (dummy_image, None),
  f"/workspace/checkpoints/baseline/nvclip_clipa_vit_h14_700M_vision.onnx",
  export_params=True,
  input_names=["IMAGE"],
  output_names=["LOGITS_PER_IMAGE"],
  opset_version=19,
  dynamic_axes={
      "IMAGE": {
          0: "image_batch_size",
      },
      "LOGITS_PER_IMAGE": {
          0: "image_batch_size",
          1: "text_batch_size",
      },
  },
  verbose=True
)


# Text model
torch.onnx.export(model, (None, text),
  f"/workspace/checkpoints/baseline/nvclip_clipa_vit_h14_700M_text.onnx",
  export_params=True,
  input_names=["TEXT"],
  output_names=["LOGITS_PER_TEXT"],
  opset_version=19,
  dynamic_axes={
      "TEXT": {
          0: "text_batch_size",
      },
      "LOGITS_PER_TEXT": {
          0: "text_batch_size",
          1: "image_batch_size",
      },
  },
  verbose=True
)
