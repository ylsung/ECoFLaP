"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import json
import logging
import os
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.distributed as dist
import webdataset as wds
from lavis.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from lavis.common.registry import registry
from lavis.common.utils import is_url
from lavis.datasets.data_utils import concat_datasets, reorg_datasets_by_split
from lavis.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import ChainDataset


@registry.register_runner("runner_base")
class RunnerBase:
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    def __init__(self, cfg, task, model, datasets, job_id=None):
        self.config = cfg
        self.job_id = job_id

        self.task = task
        self.datasets = datasets

        self._model = model

        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0

        # self.setup_seeds()

        if job_id is not None:
            self.setup_output_dir()

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)

        return self._device

    @property
    def use_distributed(self):
        return self.config.run_cfg.distributed

    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """
        # move model to device
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            # distributed training wrapper
            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = DDP(
                        self._model, device_ids=[self.config.run_cfg.gpu]
                    )
            else:
                self._wrapped_model = self._model

        return self._wrapped_model

    @property
    def optimizer(self):
        # TODO make optimizer class and configurations
        if self._optimizer is None:
            num_parameters = 0
            p_wd, p_non_wd = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue  # frozen weights
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)
                num_parameters += p.data.nelement()
            logging.info("number of trainable parameters: %d" % num_parameters)
            optim_params = [
                {
                    "params": p_wd,
                    "weight_decay": float(self.config.run_cfg.weight_decay),
                },
                {"params": p_non_wd, "weight_decay": 0},
            ]
            beta2 = self.config.run_cfg.get("beta2", 0.999)
            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.run_cfg.init_lr),
                weight_decay=float(self.config.run_cfg.weight_decay),
                betas=(0.9, beta2),
            )

        return self._optimizer

    @property
    def scaler(self):
        amp = self.config.run_cfg.get("amp", False)

        if amp:
            if self._scaler is None:
                self._scaler = torch.cuda.amp.GradScaler()

        return self._scaler

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)

            # max_epoch = self.config.run_cfg.max_epoch
            max_epoch = self.max_epoch
            # min_lr = self.config.run_cfg.min_lr
            min_lr = self.min_lr
            # init_lr = self.config.run_cfg.init_lr
            init_lr = self.init_lr

            # optional parameters
            decay_rate = self.config.run_cfg.get("lr_decay_rate", None)
            warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
            warmup_steps = self.config.run_cfg.get("warmup_steps", 0)

            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )

        return self._lr_sched

    @property
    def dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.

        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Currently do not support multiple datasets for validation and test.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """
        if self._dataloaders is None:
            # reoganize datasets by split and concatenate/chain if necessary
            dataset_ratios = self.config.run_cfg.get("train_dataset_ratios", None)

            # concatenate map-style datasets and chain wds.DataPipe datasets separately
            # training set becomes a tuple (ConcatDataset, ChainDataset), both are
            # optional but at least one of them is required. The resultant ConcatDataset
            # and ChainDataset will be sampled evenly.
            logging.info(
                "dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline)."
            )

            datasets = reorg_datasets_by_split(self.datasets)
            self.datasets = concat_datasets(datasets)

            # print dataset statistics after concatenation/chaining
            for split_name in self.datasets:
                if isinstance(self.datasets[split_name], tuple) or isinstance(
                    self.datasets[split_name], list
                ):
                    # mixed wds.DataPipeline and torch.utils.data.Dataset
                    num_records = sum(
                        [
                            len(d)
                            if not type(d) in [wds.DataPipeline, ChainDataset]
                            else 0
                            for d in self.datasets[split_name]
                        ]
                    )

                else:
                    if hasattr(self.datasets[split_name], "__len__"):
                        # a single map-style dataset
                        num_records = len(self.datasets[split_name])
                    else:
                        # a single wds.DataPipeline
                        num_records = -1
                        logging.info(
                            "Only a single wds.DataPipeline dataset, no __len__ attribute."
                        )

                if num_records >= 0:
                    logging.info(
                        "Loaded {} records for {} split from the dataset.".format(
                            num_records, split_name
                        )
                    )

            # create dataloaders
            split_names = sorted(self.datasets.keys())

            datasets = [self.datasets[split] for split in split_names]
            is_trains = [split in self.train_splits for split in split_names]

            batch_sizes = [
                self.config.run_cfg.batch_size_train
                if split == "train"
                else self.config.run_cfg.batch_size_eval
                for split in split_names
            ]

            collate_fns = []
            for dataset in datasets:
                if isinstance(dataset, tuple) or isinstance(dataset, list):
                    collate_fns.append([getattr(d, "collater", None) for d in dataset])
                else:
                    collate_fns.append(getattr(dataset, "collater", None))

            dataloaders = self.create_loaders(
                datasets=datasets,
                num_workers=self.config.run_cfg.num_workers,
                batch_sizes=batch_sizes,
                is_trains=is_trains,
                collate_fns=collate_fns,
                dataset_ratios=dataset_ratios,
            )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"

    @property
    def max_epoch(self):
        return int(self.config.run_cfg.max_epoch)

    @property
    def log_freq(self):
        log_freq = self.config.run_cfg.get("log_freq", 50)
        return int(log_freq)

    @property
    def init_lr(self):
        return float(self.config.run_cfg.init_lr)

    @property
    def min_lr(self):
        return float(self.config.run_cfg.min_lr)

    @property
    def accum_grad_iters(self):
        return int(self.config.run_cfg.get("accum_grad_iters", 1))

    @property
    def valid_splits(self):
        valid_splits = self.config.run_cfg.get("valid_splits", [])

        if len(valid_splits) == 0:
            logging.info("No validation splits found.")

        return valid_splits

    @property
    def test_splits(self):
        test_splits = self.config.run_cfg.get("test_splits", [])

        return test_splits

    @property
    def train_splits(self):
        train_splits = self.config.run_cfg.get("train_splits", [])

        if len(train_splits) == 0:
            logging.info("Empty train splits.")

        return train_splits

    @property
    def evaluate_only(self):
        """
        Set to True to skip training.
        """
        return self.config.run_cfg.evaluate

    @property
    def use_dist_eval_sampler(self):
        return self.config.run_cfg.get("use_dist_eval_sampler", True)

    @property
    def resume_ckpt_path(self):
        return self.config.run_cfg.get("resume_ckpt_path", None)

    @property
    def train_loader(self):
        train_dataloader = self.dataloaders["train"]

        return train_dataloader

    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))

        output_dir = lib_root / self.config.run_cfg.output_dir / self.job_id
        result_dir = output_dir / "result"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        self.log_config()

        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            # training phase
            if not self.evaluate_only:
                logging.info("Start training")
                train_stats = self.train_epoch(cur_epoch)
                self.log_stats(split_name="train", stats=train_stats)

            # evaluation phase
            if len(self.valid_splits) > 0:
                for split_name in self.valid_splits:
                    logging.info("Evaluating on {}.".format(split_name))

                    val_log = self.eval_epoch(
                        split_name=split_name, cur_epoch=cur_epoch
                    )
                    if val_log is not None:
                        if is_main_process():
                            assert (
                                "agg_metrics" in val_log
                            ), "No agg_metrics found in validation log."

                            agg_metrics = val_log["agg_metrics"]
                            if agg_metrics > best_agg_metric and split_name == "val":
                                best_epoch, best_agg_metric = cur_epoch, agg_metrics

                                self._save_checkpoint(cur_epoch, is_best=True)

                            val_log.update({"best_epoch": best_epoch})
                            self.log_stats(val_log, split_name)

            else:
                # if no validation split is provided, we just save the checkpoint at the end of each epoch.
                if not self.evaluate_only:
                    self._save_checkpoint(cur_epoch, is_best=False)

            if self.evaluate_only:
                break

            dist.barrier()

        # testing phase
        test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
        self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    def evaluate(self, cur_epoch="best", skip_reload=False):
        test_logs = dict()

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_logs[split_name] = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                )

            return test_logs

    def get_data_derivative(self, num_data=128, power=2, num_logits=1, vision_weight=0.0):
        model = self.unwrap_dist_model(self.model)
        model.eval()

        # record the requires_grad variable, for recovering it later
        requires_grad_record = {n: p.requires_grad for n, p in model.named_parameters()}

        dtype_record = {n: p.dtype for n, p in model.state_dict().items()}

        for n, p in model.state_dict().items():
            p.data = p.data.type(torch.bfloat16)

        # set requires_grad to be true for getting model's derivatives
        for n, p in model.named_parameters():
            p.requires_grad = True

        batch_size_train_record = self.config.run_cfg.batch_size_train
        batch_size_eval_record = self.config.run_cfg.batch_size_eval

        self.config.run_cfg.batch_size_train = 1
        self.config.run_cfg.batch_size_eval = 1

        split_name = self.test_splits[0]
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        self.config.run_cfg.batch_size_train = batch_size_train_record
        self.config.run_cfg.batch_size_eval = batch_size_eval_record

        self.task.before_evaluation(
            model=model,
            dataset=data_loader.dataset,
        )

        derivative_info = self.task.get_data_derivative(
            model=model, 
            data_loader=data_loader, 
            num_data=num_data, 
            power=power,
            num_logits=num_logits,
            vision_weight=vision_weight,
            cuda_enabled=self.cuda_enabled
        )

        # set to original requires grad
        for n, p in model.named_parameters():
            p.requires_grad = requires_grad_record[n]

        for n, p in model.state_dict().items():
            p.data = p.data.type(dtype_record[n])

        return derivative_info

    def get_activations(self, num_data=128, power=2):
        import transformers
        import torch.nn.functional as F

        model = self.unwrap_dist_model(self.model)
        model.eval()

        output_representations = defaultdict(float)
        input_representations = defaultdict(float)

        if power == 1:
            scale_method = torch.abs
        elif power == 2:
            scale_method = torch.square
        else:
            raise ValueError(f"power in `get_data_derivative` can only be 1 or 2, but got {power}")

        def hook_func(module, args, output):
            if isinstance(output, ((tuple, list))):
                output = output[0]

            if isinstance(
                output, 
                (transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions,
                transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions)
            ):
                avg_output = scale_method(output.last_hidden_state).mean(1).sum(0)
            elif isinstance(
                output,
                (transformers.modeling_outputs.Seq2SeqLMOutput)
            ):
                avg_output = scale_method(output.logits).mean(1).sum(0)
            elif getattr(output, "shape", None) is None: # used for finding if any output format are not what we want
                print(type(output))
                print(output.keys())
            elif len(output.shape) == 4:
                # if it is conv # shape = (B, D, P, P)
                avg_output = scale_method(output).mean(-1).mean(-1).sum(0)
            else:
                # shape = (B, L, D)

                avg_output = scale_method(output).mean(1).sum(0)

            # output_representations.shape = (D, )
            output_representations[module.module_name] += avg_output.detach().cpu() / num_data

            if isinstance(args, ((tuple, list))) and len(args) > 0:
                input = args[0]
            else:
                input = torch.zeros(1, dtype=avg_output.dtype)
                
            if isinstance(input, torch.LongTensor):
                input = torch.zeros(1)
                print(module.module_name)
            # add the attentino output that is not computed by module.
            if module.module_name.startswith("visual_encoder") and module.module_name.endswith("attn"):
                output = F.linear(input=input, weight=module.qkv.weight)

                output_representations[module.module_name + ".qkv"] += scale_method(output).mean(1).sum(0).detach().cpu() / num_data

            if len(input.shape) == 3: # norm input
                avg_input = scale_method(input).mean(1).sum(0)
            elif len(input.shape) == 1: # dummy input
                avg_input = input
            elif "visual_encoder.patch_embed" in module.module_name or "visual_encoder" == module.module_name:
                avg_input = scale_method(input).mean(-1).mean(-1).sum(0)
            elif "attn_drop" in module.module_name or "self.dropout" in module.module_name:
                avg_input = scale_method(input).mean(1).mean(-1).sum(0) # may not be right, but this is not used so it is fine
            elif any(n in module.module_name for n in [
                "relative_attention_bias",
                "t5_model.shared",
            ]):
                avg_input = torch.zeros(1, dtype=avg_output.dtype)
            else:
                print(module.module_name, input.shape)
                raise ValueError(f"The `{module.module_name}` module is unseen and need manually definition.")

            input_representations[module.module_name] += avg_input.detach().cpu() / num_data

            if module.module_name.startswith("visual_encoder") and module.module_name.endswith("attn"):
                input_representations[module.module_name + ".qkv"] += scale_method(input).mean(1).sum(0).detach().cpu() / num_data

        hooks = []
        for name, module in model.named_modules():
            # if any([name.endswith(n) for n in keys_to_track]):
            module.module_name = name
            hook = module.register_forward_hook(hook_func)
            hooks.append(hook)

        # doing the forward
        batch_size_train_record = self.config.run_cfg.batch_size_train
        batch_size_eval_record = self.config.run_cfg.batch_size_eval

        batch_size = 16

        self.config.run_cfg.batch_size_train = batch_size
        self.config.run_cfg.batch_size_eval = batch_size

        assert num_data % batch_size == 0, f"num_data should be dividable by {batch_size}."

        split_name = self.test_splits[0]
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        self.config.run_cfg.batch_size_train = batch_size_train_record
        self.config.run_cfg.batch_size_eval = batch_size_eval_record

        # get the activations by forwarding the data through the model
        self.task.get_activations(
            model=model, 
            data_loader=data_loader, 
            num_data=num_data, 
            cuda_enabled=self.cuda_enabled
        )

        for h in hooks:
            h.remove()

        return input_representations, output_representations

    def get_last_activations(self, num_data=128, power=2):
        import transformers
        import torch.nn.functional as F

        model = self.unwrap_dist_model(self.model)
        model.eval()

        if power == 1:
            scale_method = torch.abs
        elif power == 2:
            scale_method = torch.square
        else:
            raise ValueError(f"power in `get_data_derivative` can only be 1 or 2, but got {power}")

        # doing the forward
        batch_size_train_record = self.config.run_cfg.batch_size_train
        batch_size_eval_record = self.config.run_cfg.batch_size_eval

        batch_size = 16

        self.config.run_cfg.batch_size_train = batch_size
        self.config.run_cfg.batch_size_eval = batch_size

        assert num_data % batch_size == 0, f"num_data should be dividable by {batch_size}."

        split_name = self.test_splits[0]
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        self.config.run_cfg.batch_size_train = batch_size_train_record
        self.config.run_cfg.batch_size_eval = batch_size_eval_record

        # get the activations by forwarding the data through the model
        output = self.task.get_activations(
            model=model, 
            data_loader=data_loader, 
            num_data=num_data,
            cuda_enabled=self.cuda_enabled
        )

        return output

    def get_dataloader_for_importance_computation(self, num_data=128, power=2, batch_size=1):
        import transformers
        import torch.nn.functional as F

        if power == 1:
            scale_method = torch.abs
        elif power == 2:
            scale_method = torch.square
        else:
            raise ValueError(f"power in `get_data_derivative` can only be 1 or 2, but got {power}")

        # doing the forward
        batch_size_train_record = self.config.run_cfg.batch_size_train
        batch_size_eval_record = self.config.run_cfg.batch_size_eval

        self.config.run_cfg.batch_size_train = batch_size
        self.config.run_cfg.batch_size_eval = batch_size

        assert num_data % batch_size == 0, f"num_data should be dividable by {batch_size}."

        split_name = self.test_splits[0]
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        self.config.run_cfg.batch_size_train = batch_size_train_record
        self.config.run_cfg.batch_size_eval = batch_size_eval_record
        
        class DataLoaderWrapper:
            def __init__(self, dataloader, length):
                self.dataloader = dataloader
                self.dataset = dataloader.dataset
                self.length = min(length, len(dataloader))

            def __iter__(self):
                counter = 0
                for batch in self.dataloader:
                    if counter < self.length:
                        yield batch
                        counter += 1
                    else:
                        break

            def __len__(self):
                return self.length
            
        data_loader = DataLoaderWrapper(data_loader, num_data//batch_size)

        # get the activations by forwarding the data through the model
        return data_loader

    def convert_activation_to_importance(self, input_representations, output_representations, use_input_activation=False):
        # Visual encoder

        importance_measure = {}

        model = self.unwrap_dist_model(self.model)

        for n, p in model.named_parameters():
            
            is_weight = "weight" in n

            n_no_postfix = n.replace(".weight", "") if is_weight else n.replace(".bias", "")

            input_importance = None
            if "visual_encoder.cls_token" in n or "visual_encoder.pos_embed" in n:
                importance = output_representations["visual_encoder.pos_drop"].unsqueeze(0).unsqueeze(0)
            elif "query_tokens" in n:
                importance = output_representations["Qformer.bert.embeddings.LayerNorm"].unsqueeze(0).unsqueeze(0)
            elif "patch_embed.proj" in n:
                importance = output_representations[n_no_postfix]
                if is_weight:
                    importance = importance.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                    # only weight will use input representation
                    input_importance = input_representations[n_no_postfix]
                    input_importance = input_importance.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            elif ".shared." in n:
                # embedding layers
                importance = output_representations[n_no_postfix].unsqueeze(0)
            elif any(ln_n in n for ln_n in ["LayerNorm", "norm", "ln"]):
                importance = output_representations[n_no_postfix]
            elif "q_bias" in n:
                n_in_rep = n.replace("q_bias", "qkv")
                importance = output_representations[n_in_rep].chunk(3)[0]
            elif "v_bias" in n:
                n_in_rep = n.replace("v_bias", "qkv")
                importance = output_representations[n_in_rep].chunk(3)[-1]
            else:
                # linear weight
                importance = output_representations[n_no_postfix]
                
                if is_weight:
                    importance = importance.unsqueeze(-1)

                    # only weight will use input representation
                    input_importance = input_representations[n_no_postfix]
                    input_importance = input_importance.unsqueeze(0)

            # print(n, importance.shape, p.shape)

            importance_measure[n] = importance.expand(p.shape)

            if use_input_activation and input_importance is not None:

                # print(n, input_importance.shape, p.shape)
                input_importance = input_importance.expand(p.shape)
                importance_measure[n] = (importance_measure[n] + input_importance) / 2

            assert importance_measure[n].shape == p.shape, f"{importance_measure[n].shape} is different from {p.shape}."

        return importance_measure

    def train_epoch(self, epoch):
        # train
        self.model.train()

        return self.task.train_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
        )

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.task.evaluation(model, data_loader)

        if results is not None:
            return self.task.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
                orig_total_size=getattr(self, "orig_total_size", 0),
                distilled_total_size=getattr(self, "distilled_total_size", 0),
            )

    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """

        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            # create a single dataloader for each split
            if isinstance(dataset, ChainDataset) or isinstance(
                dataset, wds.DataPipeline
            ):
                # wds.WebdDataset instance are chained together
                # webdataset.DataPipeline has its own sampler and collate_fn
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                )
            else:
                # map-style dataset are concatenated together
                # setup distributed sampler
                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                    if not self.use_dist_eval_sampler:
                        # e.g. retrieval evaluation
                        sampler = sampler if is_train else None
                else:
                    sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=sampler is None and is_train,
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                )
                loader = PrefetchLoader(loader)

                if is_train:
                    loader = IterLoader(loader, use_distributed=self.use_distributed)

            return loader

        loaders = []

        for dataset, bsz, is_train, collate_fn in zip(
            datasets, batch_sizes, is_trains, collate_fns
        ):
            if isinstance(dataset, list) or isinstance(dataset, tuple):
                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz, is_train, collate_fn[i])
                        for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)

            loaders.append(loader)

        return loaders

    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")

        logging.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            logging.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        self.unwrap_dist_model(self.model).load_state_dict(state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        logging.info("Resume checkpoint from {}".format(url_or_filename))

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")
