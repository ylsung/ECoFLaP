"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from pathlib import Path

from omegaconf import OmegaConf

from lavis.common.utils import (
    cleanup_dir,
    download_and_extract_archive,
    get_abs_path,
    get_cache_path,
)


def download_file(url, filename):
    max_retries = 20
    cur_retries = 0

    header = headers[0]

    while cur_retries < max_retries:
        try:
            r = requests.get(url, headers=header, timeout=10)
            with open(filename, "wb") as f:
                f.write(r.content)

            break
        except Exception as e:
            logging.info(" ".join(repr(e).splitlines()))
            logging.error(url)
            cur_retries += 1

            # random sample a header from headers
            header = headers[np.random.randint(0, len(headers))]

    time.sleep(3 + cur_retries * 2)


DATA_URL = {
    "train": "http://images.cocodataset.org/zips/train2014.zip",  # md5: 0da8c0bd3d6becc4dcb32757491aca88
    "val": "http://images.cocodataset.org/zips/val2014.zip",  # md5: a3d79f5ed8d289b7a7554ce06a5782b3
    "test": "http://images.cocodataset.org/zips/test2014.zip",  # md5: 04127eef689ceac55e3a572c2c92f264
    "test2015": "http://images.cocodataset.org/zips/test2015.zip",  # md5: 04127eef689ceac55e3a572c2c92f264
}


def download_datasets(root, url):
    download_and_extract_archive(url=url, download_root=root, extract_root=storage_dir)


if __name__ == "__main__":

    config_path = get_abs_path("configs/datasets/coco/defaults_vqa.yaml")

    config = OmegaConf.load(
        config_path
    )

    print(config)
    print(type(config))

    for k, v in config.config.datasets.coco_vqa.build_info.annotations.items():

        print(k, v)

        # for url, storage_path in zip(v.url, v.storage):
        #     storage_path = Path(get_cache_path(storage_path))

        #     if storage_path.exists():
        #         print(f"File already exists at {storage_path}. Aborting.")
        #         exit(0)

        #     try:
        #         print("Downloading {} to {}".format(url, storage_path))
        #         download_file(url, storage_path)
        #     except Exception as e:
        #         # remove download dir if failed
        #         print("Failed to download file. Aborting.")
