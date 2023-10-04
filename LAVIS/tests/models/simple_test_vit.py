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


from lavis.compression import load_pruner

# config = {
#     "task_type": "vision",
#     "t5_prune_spec": None,
#     "vit_prune_spec": "40-0.5-1.0-1.0",
#     "importance_scores_cache": None,
#     "keep_indices_cache": None,
#     "is_strct_pruning": False,
#     "is_global": False,
# }

config = {
    "prune_spec": "40-0.9-1.0-1.0",
    "importance_scores_cache": None,
    "keep_indices_cache": None,
    "is_strct_pruning": False,
    "is_global": False,
    "sparsity_ratio_granularity": "layer",
}

# loads BLIP2-FLAN-T5XL caption model,
model, vis_processors, _ = load_model_and_preprocess(
    name="eva_clip", model_type="EVA-CLIP-g-336", is_eval=True, device=device
)

class DSet(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        
        from lavis.datasets.builders.imagefolder_builder import imagenet_classnames
        
        self.classnames = imagenet_classnames
        
        self.data = [
            {"image": vis_processors["eval"](raw_image).to(device),
            "text_input": "abcddd",
            "text_output": "123456",
            "label": 2,
            },
            {"image": vis_processors["eval"](raw_image).to(device),
            "text_input": "dsafdsa",
            "text_output": "134321",
            "label": 4,
            },
        ]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
        
dset = DSet()

model.classifier = model.zero_shot_classifier(
    classnames=dset.classnames,
    templates=model.prompt_templates,
)

del model.text

method = "vit_wanda_pruner"

dloader = torch.utils.data.DataLoader(dset, batch_size=1)
pruner = load_pruner(method, model, dloader, cfg=config)

total_size = sum(
    param.numel() for param in model.parameters()
)

model, _ = pruner.prune()

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# generate caption
caption = model.predict({"image": image, "label": 3})

print(caption)

# if "unstrct" in method:
if config["is_strct_pruning"] == False:
    distilled_total_size = sum(
        (param != 0).float().sum() for param in model.parameters()
    )
else:
    distilled_total_size = sum(
        param.numel() for param in model.parameters()
    )

print(distilled_total_size / total_size * 100.0)
