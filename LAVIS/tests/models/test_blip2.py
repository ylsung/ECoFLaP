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


class TestBlip2:
    def test_blip2_opt2p7b(self):
        # loads BLIP2-OPT-2.7b caption model, without finetuning on coco.
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["the merlion fountain in singapore"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

    def test_blip2_opt2p7b_coco(self):
        # loads BLIP2-OPT-2.7b caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt",
            model_type="caption_coco_opt2.7b",
            is_eval=True,
            device=device,
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["a statue of a mermaid spraying water into the air"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

    def test_blip2_opt6p7b(self):
        # loads BLIP2-OPT-2.7b caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["a statue of a merlion in front of a water fountain"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

    def test_blip2_opt6p7b_coco(self):
        # loads BLIP2-OPT-2.7b caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt",
            model_type="caption_coco_opt6.7b",
            is_eval=True,
            device=device,
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["a large fountain spraying water into the air"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

    def test_blip2_flant5xl(self):
        # loads BLIP2-FLAN-T5XL caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["marina bay sands, singapore"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3
    
    def test_blip2_flant5xl_merge(self):
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

        model.t5_model = t5_modify_with_weight_init(model.t5_model, config)

        for name, param in model.t5_model.named_parameters():
            param.requires_grad = False

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        print(caption)

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3
        
    def test_blip2_flant5xl_func_pruning(self):
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

        model.t5_model = t5_modify_with_weight_init(model.t5_model, config)

        for name, param in model.t5_model.named_parameters():
            param.requires_grad = False

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        print(caption)

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3
    
    def test_blip2_flant5xl_strct_mag_pruner(self):
        from lavis.compression import load_pruner
        
        config = {
            "task_type": "cc3m",
            "prune_models": "t5+vit",
            "t5_prune_spec": "24-1.0-0.5-0.5",
            "vit_prune_spec": "39-1.0-0.5-0.5",
            "importance_scores_cache": None,
            "keep_indices_cache": None,
            "is_strct_pruning": False,
            "is_global": False,
        }

        # loads BLIP2-FLAN-T5XL caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
        )
        
        class DSet(torch.utils.data.Dataset):
            def __init__(self):
                super().__init__()
                
                self.data = [
                    {"image": vis_processors["eval"](raw_image).unsqueeze(0).to(device),
                    "text_input": "abcddd",
                    "text_output": "123456"
                    },
                ]
                
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
                
        dset = DSet()
        
        dloader = torch.utils.data.DataLoader(dset, batch_size=1)
        pruner = load_pruner("strct_mag_pruner", model, dloader, cfg=config)
        
        total_size = sum(
            param.numel() for param in model.parameters()
        )
        
        model, _ = pruner.prune()
        
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        print(caption)

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3
        
        distilled_total_size = sum(
            param.numel() for param in model.parameters()
        )

        print(distilled_total_size / total_size * 100.0)
    
    def test_blip2_flant5xl_unstrct_mag_pruner(self):
        from lavis.compression import load_pruner
        
        config = {
            "task_type": "cc3m",
            "prune_models": "t5+vit",
            "t5_prune_spec": "24-0.5-1.0-1.0",
            "vit_prune_spec": "39-0.5-1.0-1.0",
            "importance_scores_cache": None,
            "keep_indices_cache": None,
            "is_strct_pruning": False,
            "is_global": False,
        }

        # loads BLIP2-FLAN-T5XL caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
        )
        
        class DSet(torch.utils.data.Dataset):
            def __init__(self):
                super().__init__()
                
                self.data = [
                    {"image": vis_processors["eval"](raw_image).unsqueeze(0).to(device),
                    "text_input": "abcddd",
                    "text_output": "123456"
                    },
                ]
                
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
                
        dset = DSet()
        
        dloader = torch.utils.data.DataLoader(dset, batch_size=1)
        pruner = load_pruner("unstrct_mag_pruner", model, dloader, cfg=config)
        
        total_size = sum(
            param.numel() for param in model.parameters()
        )
        
        model, _ = pruner.prune()
        
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        print(caption)

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3
        
        distilled_total_size = sum(
            (param != 0).float().sum() for param in model.parameters()
        )

        print(distilled_total_size / total_size * 100.0)


    def test_blip2_flant5xxl(self):
        # loads BLIP2-FLAN-T5XXL caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5",
            model_type="pretrain_flant5xxl",
            is_eval=True,
            device=device,
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["the merlion statue in singapore"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3
