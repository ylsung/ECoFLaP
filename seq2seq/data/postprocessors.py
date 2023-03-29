import abc
from collections import OrderedDict
import numpy as np
from .tasks import TASK_MAPPING

from transformers.trainer_utils import EvalPrediction

"""Defines functions to process the outputs to make them ready for the evaluation."""

def string_to_float(string, default=-1., **unused_kwargs):
  """Converts string to float, using default when conversion not possible."""
  try:
    return float(string)
  except ValueError:
    return default


class PostProcessor(abc.ABC): 
    """Postprocess the predictions and labels to make them suitable for
    evaluation."""
    def __init__(self, tokenizer, ignore_pad_token_for_loss, labels_list):
       self.tokenizer = tokenizer 
       self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
       self.labels_list = labels_list

    def label_processing(self, x):
        # processing for binary labels
        # from str to int
        try:
            if x not in self.labels_list:
                x = 0
            else:
                x = int(x)

        except:
            x = 0

        return x

    def process(self, preds, labels, data_info=None):
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        # print(decoded_preds)
        decoded_preds = [self.label_processing(pred.strip()) for pred in decoded_preds]
        decoded_labels = [self.label_processing(label.strip()) for label in decoded_labels]

        # print(decoded_preds)

        return decoded_preds, decoded_labels


class STSB(PostProcessor):
    def label_processing(self, x):
        # processing for binary labels
        # from str to int
        try:
            x = float(x)
        except:
            x = 2.6

        return x


class MLM(PostProcessor):
    def process(self, preds, labels, data_info=None):
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        # print(decoded_preds)
        decoded_preds = [self.label_processing(pred.strip()) for pred in decoded_preds]
        decoded_labels = [self.label_processing(label.strip()) for label in decoded_labels]

        decoded_preds, decoded_labels = self.process_inputs(decoded_preds, decoded_labels)

        assert len(decoded_preds) == len(decoded_labels)

        return decoded_preds, decoded_labels

    def process_inputs(self, preds, labels):
        # to make sure preds and labels are in same shape
        lengths = [min(len(p), len(l)) for p, l in zip(preds, labels)]

        # flatten
        new_preds = []

        for i, p in enumerate(preds):
            new_preds += p[:lengths[i]]

        new_labels = []

        for i, l in enumerate(labels):
            new_labels += l[:lengths[i]]

        return new_preds, new_labels

    def label_processing(self, x):
        # processing for binary labels
        # from str to int
        return self.tokenizer.encode(x, add_special_tokens=False)


class SquadV2(PostProcessor):

    def process(self, preds, labels, data_info=None):
        # borrowed from https://github.com/huggingface/transformers/blob/cae78c46d658a8e496a815c2ee49b9b178fb9c9a/examples/pytorch/question-answering/run_seq2seq_qa.py#L610
        # Decode the predicted tokens.
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        references = [{"id": d["id"], "answers": d["answers"]} for d in data_info]

        for i, (decoded_pred, reference) in enumerate(zip(decoded_preds, references)):
            decoded_preds[i] = {"id": reference["id"], "prediction_text": decoded_pred, "no_answer_probability": 0.0}

        return decoded_preds, references


class Winogrande(PostProcessor):
    def label_processing(self, x):
        # processing for binary labels
        # from str to int
        try:
            if x not in self.labels_list:
                x = 1
            else:
                x = int(x)

        except:
            x = 1

        return x


class SuperGLUEMultiRC(PostProcessor):
    def process(self, preds, labels, data_info=None):
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        # print(decoded_preds)
        decoded_preds = [self.label_processing(pred.strip()) for pred in decoded_preds]
        decoded_labels = [self.label_processing(label.strip()) for label in decoded_labels]

        decoded_preds = [{"idx": d["idx"], "prediction": p} for p, d in zip(decoded_preds, data_info)]

        # print(decoded_preds)

        return decoded_preds, decoded_labels


class SuperGLUERecord(PostProcessor):
    def process(self, preds, labels, data_info=None):
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        # print(decoded_preds)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        decoded_preds = [{"idx": d["idx"], "prediction_text": p} for p, d in zip(decoded_preds, data_info)]

        references = [{"idx": d["idx"], "answers": d["answers"]} for d in data_info]

        # print(decoded_preds)

        return decoded_preds, references


# class MultiRC(PostProcessor):
#     def process(self, preds, labels, data_info):
#         preds, labels = super().process(preds, labels, data_info) 
#         preds = [{"group": info["group"], "value":pred} \
#             for info, pred in zip(data_info, preds)]
#         labels = [{"group": info["group"], "value": label}\
#             for info, label in zip(data_info, labels)] 
#         return preds, labels 

# class Record(PostProcessor):
#     def process(self, preds, labels, data_info):
#         preds, labels = super().process(preds, labels, data_info) 
#         labels = [info["answers"] for info in data_info]
#         return preds, labels 


POSTPROCESSOR_MAPPING = OrderedDict(
    [
        # ('superglue-record', Record),
        # ('superglue-multirc', MultiRC)
        ('stsb', STSB),
        ('mlm', MLM),
        ('squad_v2', SquadV2),
        ('winogrande_debiased', Winogrande),
        ('superglue-multirc', SuperGLUEMultiRC),
        ('superglue-record', SuperGLUERecord),
    ]
)

class AutoPostProcessor:
    @classmethod
    def get(self, task, tokenizer, ignore_pad_token_for_loss):
        
        labels_list = TASK_MAPPING[task].labels_list
        if task in POSTPROCESSOR_MAPPING:
            return POSTPROCESSOR_MAPPING[task](tokenizer, ignore_pad_token_for_loss, labels_list)
        return PostProcessor(tokenizer, ignore_pad_token_for_loss, labels_list)
