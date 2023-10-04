"""
 Hard coded, only use validation set now
"""

import os
from collections import OrderedDict

from torch.utils.data import Dataset
from lavis.datasets.datasets.base_dataset import BaseDataset
import random
import datasets
from torch.utils.data.dataloader import default_collate


class C4Dataset(Dataset):
    def __init__(self, text_processor, split):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # self.annotation = datasets.load_dataset("allenai/c4", data_dir="en", split=split)
        # TODO Only use a subset, should create another class for this subset class
        # codes got from https://github.com/locuslab/wanda/blob/main/lib/data.py#L44
        if split == "train":
            self.annotation = datasets.load_dataset(
                'allenai/c4', 
                'allenai--c4', 
                data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, 
                split='train',
            )
        else:
            self.annotation = datasets.load_dataset(
                'allenai/c4', 
                'allenai--c4', 
                data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, 
                split='validation',
            )

        self.text_processor = text_processor

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        pass

    def __getitem__(self, index):
        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        caption = self.text_processor(ann["text"])

        split = random.randint(1, len(caption) // 2)

        prefix = caption[:split]
        output = caption[split:]

        return {"text_input": prefix, "text_output": output}
