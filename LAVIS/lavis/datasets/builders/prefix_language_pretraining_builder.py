"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from lavis.common.registry import registry

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.prefix_language_pretraining import PrefixLanguagePretrainingDataset
from lavis.datasets.datasets.laion_dataset import LaionDataset


@registry.register_builder("prefix_conceptual_caption_3m")
class PrefixConceptualCaption3MBuilder(BaseDatasetBuilder):
    train_dataset_cls = PrefixLanguagePretrainingDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/prefix_conceptual_caption/defaults_3m.yaml"
    }


@registry.register_builder("prefix_conceptual_caption_12m")
class PrefixConceptualCaption12MBuilder(BaseDatasetBuilder):
    train_dataset_cls = PrefixLanguagePretrainingDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/prefix_conceptual_caption/defaults_12m.yaml"
    }


@registry.register_builder("prefix_sbu_caption")
class PrefixSBUCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = PrefixLanguagePretrainingDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/prefix_sbu_caption/defaults.yaml"}


@registry.register_builder("prefix_vg_caption")
class PrefixVGCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = PrefixLanguagePretrainingDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/prefix_vg/defaults_caption.yaml"}


# @registry.register_builder("prefix_laion2B_multi")
# class PrefixLaion2BMultiBuilder(BaseDatasetBuilder):
#     train_dataset_cls = LaionDataset

#     DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults_2B_multi.yaml"}

#     def _download_ann(self):
#         pass

#     def _download_vis(self):
#         pass

#     def build(self):
#         self.build_processors()

#         build_info = self.config.build_info

#         datasets = dict()
#         split = "train"  # laion dataset only has train split

#         # create datasets
#         # [NOTE] return inner_datasets (wds.DataPipeline)
#         dataset_cls = self.train_dataset_cls
#         datasets[split] = dataset_cls(
#             vis_processor=self.vis_processors[split],
#             text_processor=self.text_processors[split],
#             location=build_info.storage,
#         ).inner_dataset

#         return datasets
