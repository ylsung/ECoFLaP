# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import functools
import logging
import torch 
import re
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU' 
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ["WANDB_DISABLED"] = "true"
import sys
import subprocess
import numpy as np

# logging.disable(logging.WARNING)

from typing import Optional, List

from datasets import load_dataset, load_metric, concatenate_datasets
import transformers
import evaluate
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process, get_last_checkpoint
from seq2seq.data import AutoTask
from seq2seq.data import TaskDataCollatorForSeq2Seq, DataCollatorForT5MLM
from training_args import TrainingArguments, ModelArguments, DataTrainingArguments, PETLModelArguments
from seq2seq.utils import num_parameters, save_training_config
from dataclasses import dataclass, field
from third_party.trainers.seq2seq_trainer import CustomOptSeq2SeqTrainer, _save
from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from transformers import T5Config, T5ForConditionalGeneration
from seq2seq.data import AutoPostProcessor

from seq2seq.methods import modify_model_with_petl
from seq2seq.utils.utils import parse_args
from copy import deepcopy

from seq2seq.methods.distillation.modify_model_with_distillation import t5_modify_with_distillation
from third_party.modeling_utils import save_pretrained


logger = logging.getLogger(__name__)


def run_command(command):
    output = subprocess.getoutput(command)
    return output


TASK_TO_METRICS = {"mrpc": ["accuracy", "f1"],
                  "cola": ['matthews_correlation'],
                  "stsb": ['pearson', 'spearmanr'],
                  'sst2': ['accuracy'],
                  "mnli": ["accuracy"],
                  "mnli_mismatched": ["accuracy"],
                  "mnli_matched": ["accuracy"],
                  "qnli": ["accuracy"],
                  "rte": ["accuracy"],
                  "wnli": ["accuracy"],
                  "qqp": ["accuracy", "f1"],
                  "superglue-boolq": ["accuracy"],
                  "superglue-rte": ["accuracy"],
                  "superglue-cb": ["f1_multiclass", "accuracy"],
                  "superglue-copa": ["accuracy"],
                  "superglue-multirc": ["f1", "em"],
                  "superglue-wic": ["accuracy"],
                  "superglue-wsc.fixed": ["accuracy"],
                  "superglue-record": ["f1", "em"]
         }


def main():
    
    model_args, data_args, training_args, petl_args = \
        parse_args([ModelArguments, DataTrainingArguments, TrainingArguments, PETLModelArguments])

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print("#### last_checkpoint ", last_checkpoint)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            '''
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
            '''
            pass 
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files in the summarization task, this script will use the first column for the full texts and the
    # second column for the summaries (unless you specify column names for this with the `text_column` and
    # `summary_column` arguments).
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = T5Config.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    data_args.dataset_name = [data_args.task_name]
    data_args.eval_dataset_name = [data_args.eval_dataset_name]
    data_args.test_dataset_name = [data_args.test_dataset_name]
    data_args.dataset_config_name = [data_args.dataset_config_name]
    data_args.eval_dataset_config_name = [data_args.eval_dataset_config_name]
    data_args.test_dataset_config_name = [data_args.test_dataset_config_name]
    assert len(data_args.dataset_name) == len(data_args.dataset_config_name)
    if data_args.eval_dataset_name is not None:
        assert len(data_args.eval_dataset_name) == len(data_args.eval_dataset_config_name)
    if data_args.test_dataset_name is not None:
        assert len(data_args.test_dataset_name) == len(data_args.test_dataset_config_name)

    # Temporarily set max_target_length for training.
    #max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    
    def preprocess_function(examples, max_target_length):

        model_inputs = tokenizer(examples['source'], max_length=data_args.max_source_length,
                                padding=padding, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['target'], max_length=max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        # model_inputs["extra_fields"] = examples['extra_fields']

        return model_inputs

    def preprocess_function_t5_mlm(examples, *args, **kwargs):
        # just extract the source, and the data collator will do the rest of work
        # return {"source": examples["source"], "extra_fields": examples['extra_fields']}

        return {"source": examples["source"]}

    preprocess_function_chosen = preprocess_function_t5_mlm

    # column_names = ['source', 'target', 'extra_fields']
    column_names = ['source', 'target']
    performance_metrics = {}

    if training_args.do_train:
        train_datasets = [AutoTask.get(dataset_name, dataset_config_name,
            seed=training_args.data_seed).get(
            split="train", 
            split_validation_test=False,
            add_prefix=False,
            n_obs=data_args.max_train_samples)
            for dataset_name, dataset_config_name\
            in zip(data_args.dataset_name, data_args.dataset_config_name)]
        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(\
            tokenizer=tokenizer, default_max_length=data_args.max_target_length)\
            for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)]
        for i, train_dataset in enumerate(train_datasets):
            train_datasets[i] = train_datasets[i].map(
                functools.partial(preprocess_function_chosen, max_target_length=max_target_lengths[i]),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names, # if train_dataset != "superglue-record" else column_names+["answers"],
                load_from_cache_file=not data_args.overwrite_cache,
            )
        train_dataset = concatenate_datasets(train_datasets)
   
    if training_args.do_eval:
        eval_datasets = {eval_dataset: AutoTask.get(eval_dataset, eval_dataset_config,
            seed=training_args.data_seed).get(
            split="validation" if eval_dataset == "c4" else "test", 
            split_validation_test=False,
            add_prefix=False,
            n_obs=data_args.max_val_samples)
            for eval_dataset, eval_dataset_config in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)}
        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length( \
            tokenizer=tokenizer, default_max_length=data_args.max_target_length) \
            for dataset_name, dataset_config_name in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)]
        for k, name in enumerate(eval_datasets):
            eval_datasets[name] = eval_datasets[name].map(
                    functools.partial(preprocess_function_chosen, max_target_length=max_target_lengths[k]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names, # if name != "superglue-record" else column_names+["answers"],
                    load_from_cache_file=not data_args.overwrite_cache,
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    training_args.remove_unused_columns = False # avoid removing example["source"] when feeding examples to data collator
    data_collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=petl_args.mlm_ratio,
        mean_noise_span_length=3,
        input_length=data_args.max_source_length,
        target_length=data_args.max_target_length,
        pad_token_id=config.pad_token_id,
        decoder_start_token_id=config.decoder_start_token_id,
    )

    from seq2seq.metrics import metrics
    # just to avoid error happens, whatever metric is used doesn't effect the selected models in distillation
    metric = [AutoTask.get("mlm", "mlm").load_metric() \
        for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)][0]

    # # Extracts the extra information needed to evaluate on each dataset.
    # # These information are only used in the compute_metrics.
    # # We will assume that the test/eval dataloader does not change the order of 
    # # the data.

    # if training_args.do_train:
    #     data_info = {"eval": eval_datasets[data_args.eval_dataset_name[0]]['extra_fields'],
    #                  "test": test_datasets[data_args.test_dataset_name[0]]['extra_fields'], 
    #                  "train": train_dataset['extra_fields']}
    # else:
    #     data_info = {"eval": eval_datasets[data_args.eval_dataset_name[0]]['extra_fields'],
    #                  "test": test_datasets[data_args.test_dataset_name[0]]['extra_fields']}

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        post_processor = AutoPostProcessor.get("mlm", tokenizer,
                                               data_args.ignore_pad_token_for_loss)
        decoded_preds, decoded_labels = post_processor.process(preds, labels)
    

        # result = {}
        # for metric in eval_metrics:
        #     result.update(metric(decoded_preds, decoded_labels))

        # return result

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result["combined_score"] = np.mean(list(result.values())).item()

        return result
        
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    teacher = T5ForConditionalGeneration.from_pretrained(
        petl_args.teacher_model_name,
    )

    model.resize_token_embeddings(len(tokenizer))
    teacher.resize_token_embeddings(len(tokenizer))

    model = modify_model_with_petl(model, petl_args)

    model = t5_modify_with_distillation(model, teacher, petl_args)

    if model_args.weight_to_load is not None:
        weight_to_load = torch.load(model_args.weight_to_load, map_location="cpu")
        model.load_state_dict(weight_to_load)

    # determine trainable parameters
    for p_name, param in model.named_parameters():
        if re.fullmatch(petl_args.trainable_param_names, p_name) and "teacher" not in p_name:
            param.requires_grad = True
            print(p_name)
        else:
            param.requires_grad = False

    if training_args.parameters_with_larger_lr is not None:
        trainer_cls = CustomOptSeq2SeqTrainer
    else:
        trainer_cls = Seq2SeqTrainer

    # Initialize our Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=list(eval_datasets.values())[0] if training_args.do_eval else None,
        # data_info=data_info,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        # evaluation_metrics = TASK_TO_METRICS[data_args.dataset_name[0]]
    )
    # model.save_pretrained = functools.partial(save_pretrained, model)
    trainer._save = functools.partial(_save, trainer)

    # Saves training config. 
    if trainer.is_world_process_zero():
        os.makedirs(training_args.output_dir, exist_ok=True)

        config_to_save = {
            **vars(petl_args), 
            **vars(training_args), 
            **vars(model_args), 
            **vars(data_args),
        }

        def is_serializable(v):
            if isinstance(v, (str, int, float, bool, list, dict, set, type(None))):
                return True
            else:
                return False

        config_to_save = {
            k: v for k, v in config_to_save.items() if is_serializable(v)
        }

        save_training_config(config_to_save, training_args.output_dir)

    if training_args.print_num_parameters:
        total_trainable_params_percent = num_parameters(model, petl_args)
        trainer.save_metrics("updated_params", {"updated_params": total_trainable_params_percent})

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if training_args.compute_time:
            torch.cuda.synchronize()  # wait for move to complete
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        if training_args.compute_time:
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end)/(1000*60)
            performance_metrics.update({"total_time in minutes ": total_time})
        
        trainer.save_model()  # Saves the tokenizer too for easy upload
        train_metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        train_metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        print(
            "Memory utilization",
            peak_memory,
            "GB"
        )
        performance_metrics.update({"peak_memory": peak_memory})

    if training_args.compute_memory or training_args.compute_time:
        print(performance_metrics)
        trainer.save_metrics("performance", performance_metrics)
    
    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        for task, eval_dataset in eval_datasets.items():
            metrics = trainer.evaluate(eval_dataset=eval_dataset,
               max_length=data_args.val_max_target_length, num_beams=training_args.generation_num_beams,
            )
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        # only useful when computing inference memory
        if torch.cuda.is_available() and training_args.compute_memory:
            peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
            print(
                "Memory utilization",
                peak_memory,
                "GB"
            )


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
