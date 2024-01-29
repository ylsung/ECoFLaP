import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        
        if cfg.PRUNER.METHOD is not None:
            from copy import deepcopy
            from dassl.data import DataManager
            
            pruning_cfg = deepcopy(cfg)
            pruning_cfg.defrost()
            pruning_cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 16
            dm = DataManager(pruning_cfg)
            
            if cfg.PRUNER.METHOD == "wanda":
                from .pruners.wanda_pruner import CLIPLayerWandaPruner
                import torch.nn.functional as F
                
                pruner = CLIPLayerWandaPruner(model=clip_model, data_loader=dm.train_loader_x,
                                              **dict(cfg.PRUNER.PARAMS)
                                              )
                
            elif cfg.PRUNER.METHOD == "sparsegpt":
                from .pruners.sparsegpt_pruner import CLIPLayerSparseGPTPruner
                import torch.nn.functional as F
                
                pruner = CLIPLayerSparseGPTPruner(model=clip_model, data_loader=dm.train_loader_x,
                                              **dict(cfg.PRUNER.PARAMS)
                                              )
            
            def forward_to_cache(model, batch, device):
                image = batch["img"]
                label = batch["label"]
                
                text = [temp.format(classnames[l.item()].replace("_", " ")) for l in label]
                text = torch.cat([clip.tokenize(t) for t in text])
                image = image.to(device)
                text = text.to(device)
                
                logits_per_image, logits_per_text = model(image, text)

                ground_truth = torch.arange(len(logits_per_image)).long().to(device)
    
                loss = (
                    F.cross_entropy(logits_per_image.float(), ground_truth)
                    + F.cross_entropy(logits_per_text.float(), ground_truth)
                ) / 2
                
                return loss, label.shape[0]
            
            pruner.forward_to_cache = forward_to_cache
                
            clip_model, sparsity_dict = pruner.prune()
            
            if sparsity_dict is not None:
                import json
                import os
                with open(os.path.join(cfg.OUTPUT_DIR, "sparsity_dict.json"), "w") as f:
                    json.dump(sparsity_dict, f, indent=4) 

            total_params = 0
            remaining_params = 0
            
            for n, p in clip_model.named_parameters():
                total_params += p.numel()
                remaining_params += (p != 0).float().sum()
                
            print(f"Sparsity: {1 - remaining_params/total_params: .2f}")

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """Prompt ensembling."""

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model
