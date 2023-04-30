"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
from torchvision import datasets


class CIFAR100Dataset(BaseDataset):
    def __init__(self, vis_processor, vis_root, is_train, **kwargs):
        super().__init__(vis_processor=vis_processor, vis_root=vis_root)

        self.inner_dataset = datasets.CIFAR100(vis_root, is_train, download=True)

        self.annotation = [
            {"label": d[1]}
            for i, d in enumerate(self.inner_dataset)
        ]

        self.classnames = self.inner_dataset.classes

        self._add_instance_ids()

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image = self.inner_dataset[index][0]
        image = self.vis_processor(image)

        return {
            "image": image,
            "label": ann["label"],
            "instance_id": ann["instance_id"],
        }

    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["instance_id"],
                "label": self.classnames[ann["label"]],
                "image": sample["image"],
            }
        )
