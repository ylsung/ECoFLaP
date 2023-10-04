"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask

from lavis.datasets.data_utils import prepare_sample


@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass

    def get_data_derivative(self, model, data_loader, num_data=128, power=2, num_logits=1, vision_weight=0.0, cuda_enabled=False):
        gradients_dict = {}

        if power == 1:
            grad_method = torch.abs
        elif power == 2:
            grad_method = torch.square
        else:
            raise ValueError(f"power in `get_data_derivative` can only be 1 or 2, but got {power}")

        for name, param in model.named_parameters():
            gradients_dict[name] = 0

        idx = 0

        no_grad_list = set()

        for samples in data_loader:
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            loss_dict = model.forward_with_vision_auxloss(samples)
            loss = loss_dict["loss"] + loss_dict["vision_auxloss"] * vision_weight
            loss.backward()

            for name, param in model.named_parameters():

                if param.grad is not None:
                    gradients_dict[name] += grad_method(param.grad.cpu().data) / num_data
                else:
                    no_grad_list.add(name)
            
            model.zero_grad()

            idx += 1

            if idx >= num_data:
                break

        for k in no_grad_list:
            print(f"{k} has no grad")

        return gradients_dict
