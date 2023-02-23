import transformers
import datasets
from typing import Callable, List, Mapping
import functools

dset = datasets.load_dataset("glue", "rte", split="train")



def seq2seq_format(sources: List[str],
                    targets: List[str],
                    add_prefix: bool=False,
                    prefix: str=None,
                    extra_fields={}):
        src_prefix = "rte"
        sources = [src_prefix]+sources if add_prefix else sources
        return {'source': ' '.join(sources),
                'target': ' '.join(targets),
                'task': "rte"}

def preprocessor(example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return seq2seq_format(src_texts, tgt_texts, add_prefix)


print(dset.column_names)

dset.map(functools.partial(preprocessor, add_prefix=False),
                           remove_columns=dset.column_names)