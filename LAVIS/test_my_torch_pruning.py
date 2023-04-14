"""
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

Integration tests for BLIP2 models.
"""

import pytest
import torch
from lavis.models import load_model, load_model_and_preprocess
from PIL import Image

# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load sample image
raw_image = Image.open("docs/_static/merlion.png").convert("RGB")


from lavis.compression.modify_model_with_weight_init import t5_modify_with_weight_init

class Config:
    side_pretrained_weight = "3-2048"
    distillation_init = "sum"
    distilled_block_ids = "[[0,1,2],[3,4],5]"
    distilled_block_weights = None
    modules_to_merge = ".*|.*"
    permute_before_merge = False
    permute_on_block_before_merge = True

config = Config()
# loads BLIP2-FLAN-T5XL caption model,
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
)


for n, p in model.named_parameters():
    print(n, p.shape)

# model.t5_model = t5_modify_with_weight_init(model.t5_model, config)

# for name, param in model.t5_model.named_parameters():
#     param.requires_grad = False

# preprocess the image
# vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# generate caption
caption = model.generate({"image": image})

print(caption)

# generate multiple captions
captions = model.generate({"image": image}, num_captions=3)

assert len(captions) == 3