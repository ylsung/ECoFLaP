"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import logging

import numpy as np
import torch
from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("multimodal_classification")
class MultimodalClassificationTask(BaseTask):
    def __init__(self):
        super().__init__()

    def valid_step(self, model, samples):
        results = []

        outputs = model.predict(samples)

        predictions = outputs["predictions"]
        targets = outputs["targets"]

        predictions = predictions.max(1)[1].cpu().numpy()
        targets = targets.cpu().numpy()

        indices = samples[self.inst_id_key]

        for pred, tgt, index in zip(predictions, targets, indices):
            if isinstance(index, torch.Tensor):
                index = index.item()

            results.append(
                {
                    self.inst_id_key: index,
                    "prediction": pred.item(),
                    "target": tgt.item(),
                }
            )

        return results

    def get_samples_probs(self, model, samples, num_logits=1):
        outputs = model.predict(samples)

        logits = outputs["predictions"] / 100 # canceled out the multiplication by 100 in model.predict()
        targets = outputs["targets"]

        probs = torch.nn.functional.softmax(logits, -1)

        batch_size = torch.arange(len(targets)).to(targets.device)

        probs = probs[batch_size, targets].unsqueeze(-1)

        return probs

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate=self.inst_id_key,
        )

        self.orig_total_size = kwargs["orig_total_size"]
        self.distilled_total_size = kwargs["distilled_total_size"]

        metrics = self._report_metrics(
            eval_result_file=eval_result_file, split_name=split_name
        )

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        results = json.load(open(eval_result_file))

        predictions = np.array([res["prediction"] for res in results])
        targets = np.array([res["target"] for res in results])

        accuracy = (targets == predictions).sum() / targets.shape[0]
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        metrics["orig_size"] = f"{self.orig_total_size / 10 ** 9:.3f} B"
        metrics["dist_size"] = f"{self.distilled_total_size / 10 ** 9:.3f} B"

        log_stats = {split_name: {k: v for k, v in metrics.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        logging.info(metrics)
        return metrics
