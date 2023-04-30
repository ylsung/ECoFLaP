"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.cifar100_dataset import CIFAR100Dataset

import torchvision


@registry.register_builder("cifar100")
class CIFAR100Builder(BaseDatasetBuilder):
    train_dataset_cls = CIFAR100Dataset
    eval_dataset_cls = CIFAR100Dataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cifar100/defaults.yaml"}

    def _download_data(self):
        build_info = self.config.build_info
        storage_path = build_info.get(self.data_type).storage

        datasets = dict()
        for split in build_info.splits:
            if split == "train":
                is_train = True
            else:
                is_train = False

            torchvision.datasets.CIFAR100(storage_path, is_train, download=True)

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        vis_info = build_info.get(self.data_type)

        datasets = dict()
        for split in build_info.splits:
            assert split in [
                "train",
                "val",
            ], "Invalid split name {}, must be one of 'train', 'val' and 'test'."

            is_train = split == "train"

            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                vis_root=vis_info.storage,
                is_train=is_train,
            )

        return datasets
