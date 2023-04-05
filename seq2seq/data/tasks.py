from collections import OrderedDict
import collections 
import abc
import functools
from typing import Callable, List, Mapping
from utils import pad_punctuation
from seq2seq.metrics import metrics
from .utils import round_stsb_target
import datasets
import logging
import numpy as np
import torch
import re

import evaluate

logger = logging.getLogger(__name__)

class AbstractTask(abc.ABC):
    name = NotImplemented
    config = NotImplemented
    prefix = NotImplemented
    preprocessor: Callable = NotImplemented
    metric = NotImplemented
    metric_names = None
    split_map = None
    labels_list = None
    split_to_data_split: Mapping[str, str] = \
        {"train": "train", "validation": "validation", "test": "test"}
    small_datasets_without_all_splits = ["cola", "wnli", "rte", "superglue-cb", "superglue-copa", "superglue-multirc",
                                         "superglue-wic", "superglue-wsc.fixed", "superglue-rte", "mrpc", "stsb",
                                         "superglue-boolq", "winogrande", "winogrande_debiased"]
    large_data_without_all_splits = ["qqp", "qnli", "superglue-record", "sst2", "squad_v2"]
    
    def __init__(self, config, seed=42):
        self.config = config
        self.seed = seed

    def get_desired_metric_names(self):
        return self.metric_names

    def get_max_target_length(self, tokenizer, default_max_length):
        if self.labels_list is not None:
            return max([len(tokenizer.encode(label)) for label in self.labels_list])
        return default_max_length

    def seq2seq_format(self, sources: List[str],
                       targets: List[str],
                       add_prefix: bool=False,
                       prefix: str=None,
                       extra_fields={}):
        src_prefix = self.name if prefix is None else prefix
        sources = [src_prefix]+sources if add_prefix else sources
        extra_fields["task"] = self.name
        return {'source': ' '.join(sources),
                'target': ' '.join(targets),
                # 'task': self.name,
                'extra_fields': extra_fields,
                }

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs
   
    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)

        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset, n_obs=None, indices=None):
        """
        Given a dataset returns the subsampled dataset.
        :param n_obs: the number of samples of the subsampled dataset.
        :param indices: indices to select the samples from, if not given, indices are computed
        from by shuffling the given dataset.
        :return: subsampled dataset.
        """
        num_samples = len(dataset)
        n_obs = self.check_n_obs(n_obs, num_samples)
        if indices is None:
           indices = self.shuffled_indices(dataset)
        indices = indices[:n_obs]
        return dataset.select(indices)

    def load_dataset(self, split: int):
        return datasets.load_dataset(self.name, self.config, split=split)

    def get_split_indices(self, split, dataset, validation_size):
        indices = self.shuffled_indices(dataset)
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]
        
    def map_dataset(self, dataset, add_prefix):
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
                           remove_columns=dataset.column_names)

    def get(self, split, add_prefix=True, n_obs=None, split_validation_test=False):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if split_validation_test and self.name in self.small_datasets_without_all_splits \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_split_indices(split, dataset, validation_size=len(dataset)//2)
            dataset = self.subsample(dataset, n_obs, indices)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and self.name in self.large_data_without_all_splits \
                and split != "test":
            dataset = self.load_dataset(split="train")
            indices = self.get_split_indices(split, dataset, validation_size=1000)
            dataset = self.subsample(dataset, n_obs, indices)
        else:
            mapped_split = self.split_to_data_split[split]
            dataset = self.load_dataset(split=mapped_split)
            # shuffles the data and samples it.
            if n_obs is not None:
                dataset = self.subsample(dataset, n_obs)
        return self.map_dataset(dataset, add_prefix)    

class Squad(AbstractTask):
    name = "squad"
    metric = [metrics.squad]

    def load_dataset(self, split):
        return datasets.load_dataset(self.name, split=split)

    def preprocessor(self, example, add_prefix):
        answer = pad_punctuation(example['answers']['text'][0])
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer]
        return self.seq2seq_format(source, target, add_prefix)


class SquadV2(AbstractTask):
    name = "squad_v2"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    metric_names = ["exact", "f1"]

    def load_dataset(self, split):
        return datasets.load_dataset(self.name, split=split)

    def load_metric(self):
        return evaluate.load(self.name)

    def preprocessor(self, example, add_prefix):
        
        if len(example['answers']['text']) == 0:
            answer = ""
        else:
            answer = pad_punctuation(example['answers']['text'][0])

        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer]

        extra_fields = {"answers": example["answers"], "id": example["id"]}

        return self.seq2seq_format(source, target, add_prefix, extra_fields=extra_fields)


class Winogrande(AbstractTask):
    name = "winogrande_debiased"
    labels_list = ['1', '2']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset("winogrande", self.name, split=split)

    def load_metric(self):
        return evaluate.load("accuracy")

    def preprocessor(self, example, add_prefix=True):

        # sentence (string)	option1 (string)	option2 (string)
        src_texts = ["sentence:", example["sentence"], 
                     "option1:", example["option1"],
                     "option2:", example["option2"]]
        tgt_texts = [str(example["answer"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MLM(AbstractTask):
    name = "mlm"
    metric = metrics.accuracy

    def load_metric(self):
        return evaluate.load("accuracy")


class C4(AbstractTask):
    name = "allenai/c4"
    metric = metrics.accuracy

    def load_metric(self):
        return evaluate.load("accuracy")

    def load_dataset(self, split):
        return datasets.load_dataset(self.name, 'en', split=split, cache_dir="/nas-ssd/ylsung/.cache/huggingface/", num_proc=16)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['text']]
        tgt_texts = ["dummy"]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class GLUE(AbstractTask):
    benchmark_name = "glue"

    def load_dataset(self, split):
        return datasets.load_dataset(self.benchmark_name, self.name, split=split)

    def load_metric(self):
        return evaluate.load(self.benchmark_name, self.name)
 

class MRPC(GLUE):
    name = "mrpc"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score_with_invalid, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class COLA(GLUE):
    name = "cola"
    labels_list = ["0", "1"]
    metric = [metrics.matthews_corrcoef]
    metric_names = ["matthews_correlation"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SST2(GLUE):
    name = "sst2"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class STSB(GLUE):
    name = "stsb"
    labels_list = [str(np.round(label, decimals=1)) for label in np.arange(0, 5.2, 0.2)]
    metric = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]
    metric_names = ["pearson", "spearmanr"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(round_stsb_target(example['label']))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QQP(GLUE):
    name = "qqp"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score_with_invalid, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question1:", example['question1'],
                     "question2:", example["question2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MNLI(GLUE):
    name = "mnli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QNLI(GLUE):
    name = "qnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example['question'],
                     "sentence:", example["sentence"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class RTE(GLUE):
    name = "rte"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class WNLI(GLUE):
    name = "wnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SUPERGLUE(AbstractTask):
    benchmark_name = "super_glue"

    def load_dataset(self, split):
        return datasets.load_dataset(self.benchmark_name, self.name.split("-")[-1], split=split)

    def load_metric(self):
        return evaluate.load(self.benchmark_name, self.name.split("-")[-1])
 

class SuperGLUEBoolQ(SUPERGLUE):
    name="superglue-boolq"
    labels_list = ['0', '1']
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example["question"], "passage:", example["passage"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUERTE(SUPERGLUE):
    name="superglue-rte"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUECB(SUPERGLUE):
    name = "superglue-cb"
    labels_list = ['0', '1', '2']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.mean_multiclass_f1(num_classes=3), metrics.accuracy]
    metric_names = ["f1", "accuracy"]

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"], "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUECOPA(SUPERGLUE):
    name = "superglue-copa"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"] 

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"], 
                     "choice1:", example["choice1"],
                     "choice2:", example["choice2"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUEMultiRC(SUPERGLUE):
    name = "superglue-multirc"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.multirc_f1_over_all_answers,
              metrics.mean_group_metric(metrics.exact_match)]
    metric_names = ["f1_a", "exact_match"]

    def remove_markup(self, text):
        """Removes the HTML markup."""
        text = re.sub('<br>', ' ', text)
        text = re.sub('<(/)?b>', '', text)
        return text 

    def preprocessor(self, example, add_prefix=True):
        group = example['idx']['question']
        # T5 applies remove_markup to the joined string, but this should not make 
        # any difference as well.
        # https://github.com/google-research/text-to-text-transfer-transformer/blob/a1352e625db7ec114062f99d99b0565b9e45c155/t5/data/preprocessors.py#L797 
        src_texts = ["question:", self.remove_markup(example["question"]),
                     "answer:", self.remove_markup(example["answer"]),
                     "paragraph:", self.remove_markup(example["paragraph"])]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, extra_fields={"idx": example["idx"]})

   

class SuperGLUEWIC(SUPERGLUE):
    name = "superglue-wic"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"] 

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example["sentence1"],
                     "sentence2:", example["sentence2"],
                     "word:", example["word"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUEWSCFixed(SUPERGLUE):
    # source: https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py
    """Convert WSC examples to text2text format.
     WSC includes a sentence along with 2 'spans': the first denoting a noun and
     the other a pronoun. The 'label' specifies whether or not the pronoun is
     referencing the noun. This preprocessor puts ' * ' around the noun and ' # '
     around the pronoun.
     For example, a typical example from WSC might look like
     {
         'text': 'This is a test sentence .',
         'span1_text': 'test',
         'span1_index': 3,
         'span2_text': 'This',
         'span2_index': 0,
         'label': 0
     }
     This example would be transformed to
     {
         'inputs': 'wsc text: # This # is a * test * sentence .',
         'targets': 'False'
     }
    """
    name = "superglue-wsc.fixed"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"] 

    def _mark_span(self, text, span_str, span_idx, mark):
        pattern_tmpl = r'^((?:\S+\s){N})(W)'
        pattern = re.sub('N', str(span_idx), pattern_tmpl)
        pattern = re.sub('W', span_str, pattern)
        return re.sub(pattern, r'\1{0} \2 {0}'.format(mark), text)

    def preprocessor(self, example, add_prefix=True):
        # converts text as done in T5.
        text = example['text']
        text = self._mark_span(text, example['span1_text'], example['span1_index'], '*')
        # Compensate for 2 added "words" added in previous step.
        span2_index = example['span2_index'] + 2 * int(example['span1_index'] < example['span2_index'])
        text = self._mark_span(text, example['span2_text'], span2_index, '#')
        src_texts = ["text:", text]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUERecord(SUPERGLUE):
    """Convert ReCoRD examples to text2text examples.
    ReCoRD contains a passage, query containing a '@placeholder' string, and a set
    of entities that are the possible values of the placeholder. Each train and
    validation example will have a list of answers, any of which would be
    considered correct.
    For example, a typical example from ReCoRD might look like
    {
      'passsage': 'This is the passage.',
      'query': 'A @placeholder is a bird.',
      'entities': ['penguin', 'potato', 'pigeon'],
      'answers': ['penguin', 'pigeon'],
    }
    which this preprocessor would turn into the following two examples:
    {
      'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                'potato, pigeon passage: This is the passage.',
      'targets': 'penguin',
    }
    and
    {
      'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                'potato, pigeon passage: This is the passage.',
      'targets': 'pigeon',
    }
    """
    name = "superglue-record"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.squad]
    metric_names = ["exact_match", "f1"] 
    
    def preprocessor(self, batch, add_prefix=True):
        new_batch = collections.defaultdict(list)
        keys = batch.keys()
        for values in zip(*batch.values()):
            ex = {k: v for k, v in zip(keys, values)}
            # updates the passage.
            passage = ex['passage']
            passage = re.sub(r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
            passage = re.sub(r'\n@highlight\n', '. ', passage)
            inputs = f"record query: {ex['query']} entities: {', '.join(ex['entities'])} passage: {passage}"
            if add_prefix:
                inputs = self.name + " " + inputs 
            # duplicates the samples based on  number of answers.
            num_answers = len(ex["answers"])
            new_batch["source"].extend([inputs]) 
            new_batch["target"].extend([ex["answers"][0]] if num_answers > 0 else ["<unk>"])
            new_batch["extra_fields"].extend([{"idx": ex["idx"], "answers": ex["answers"], "task": self.name}]) 
        return new_batch
    
    def map_dataset(self, dataset, add_prefix=True):
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix), 
            batched=True, remove_columns=dataset.column_names)


TASK_MAPPING = OrderedDict(
    [
        ('squad', Squad),
        ('squad_v2', SquadV2),
        ('winogrande_debiased', Winogrande),
        ('mrpc', MRPC),
        ('cola', COLA),
        ('sst2', SST2),
        ('qnli', QNLI),
        ('rte', RTE),
        ('wnli', WNLI),
        ('mnli', MNLI),
        ('qqp', QQP),
        ('stsb', STSB),
        ('superglue-boolq', SuperGLUEBoolQ),
        ('superglue-rte', SuperGLUERTE),
        ('superglue-cb', SuperGLUECB),
        ('superglue-copa', SuperGLUECOPA),
        ('superglue-multirc', SuperGLUEMultiRC),
        ('superglue-wic', SuperGLUEWIC),
        ('superglue-wsc.fixed', SuperGLUEWSCFixed),
        ('superglue-record', SuperGLUERecord),
        ('mlm', MLM),
        ('c4', C4)
    ]
)

class AutoTask:
    @classmethod
    def get(self, task, config, seed=42):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](config, seed)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
