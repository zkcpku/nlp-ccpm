import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union

import datasets
import numpy as np
import torch
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version

@dataclass
class DataBinaryCls:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        labels = [feature.pop('labels') for feature in features]
        batch_size = len(features)
        # num_choices = len(features[0]["input_ids"])
        flattened_features = [{k:v for k,v in feature.items()} for feature in features]

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


# Preprocessing the datasets.
def get_preprocess_function(max_seq_length, tokenizer, pad_to_max_length = True):
    def preprocess_function(examples, idxes):
        # print(len(examples['translation']))
        # import pdb;pdb.set_trace()
        
        def split_string(c,l = 5):
            if l == 5:
                return [c[:2],c[2:5]]
            elif l == 7:
                return [c[:2],c[2:4],c[4:]]
            elif l == 12:
                return [c[:2],c[2:5],c[6:8],c[8:11]]
            elif l == 16:
                return [c[:2],c[2:4],c[4:7],c[8:10],c[10:12],c[12:15]]
        def generate_translation_and_classic_span(examples, idxes):
            '''
            7 行人初上木兰舟
            16 绿玉觜攒鸡脑破，玄金爪擘兔心开。
            5 清晨西北转
            12 篱落深村路，闾阎处士家。
            '''
            translation = []
            classic_span = []
            labels = []
            origin_idx = []
            for i in range(len(examples['choices'])):
                this_choices = []
                choice_len = len(examples['choices'][i][0])
                for c in examples['choices'][i]:
                    this_choices.extend(split_string(c,choice_len))
                
                if 'answer' in examples:
                    this_answer = examples['answer'][i]
                    this_answer = examples['choices'][i][this_answer]
                    true_spans = split_string(this_answer, choice_len)

                    this_choices = list(set(this_choices))
                    this_labels = [int(e in true_spans) for e in this_choices]

                    this_translations = [examples['translation'][i]] * len(this_choices)
                    this_origin_idx = [idxes[i]] * len(this_choices)

                    translation.extend(this_translations)
                    classic_span.extend(this_choices)
                    labels.extend(this_labels)
                    origin_idx.extend(this_origin_idx)
                else:
                    this_answer = 0
                    this_answer = examples['choices'][i][this_answer]
                    true_spans = split_string(this_answer, choice_len)

                    this_choices = list(set(this_choices))
                    this_labels = [0 for e in this_choices]

                    this_translations = [examples['translation'][i]] * len(this_choices)
                    this_origin_idx = [idxes[i]] * len(this_choices)

                    translation.extend(this_translations)
                    classic_span.extend(this_choices)
                    labels.extend(this_labels)
                    origin_idx.extend(this_origin_idx)
            return translation, classic_span, labels, origin_idx


        translation, classic_span, labels, origin_idx = generate_translation_and_classic_span(examples,idxes)

        # Flatten out
        first_sentences = translation
        second_sentences = classic_span

        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length" if pad_to_max_length else False,
        )
        # print(tokenized_examples.keys())
        results = {}
        results.update(tokenized_examples)
        # results['translation'] = translation
        # results['classic_poetry'] = classic_span
        results['origin_idx'] = origin_idx
        results['labels'] = labels
        return results 
    return preprocess_function


def unit_test():
    tokenizer = AutoTokenizer.from_pretrained('ethanyt/guwenbert-base')
    data_collator = DataBinaryCls(tokenizer=tokenizer, pad_to_multiple_of=8)
    preprocess_function = get_preprocess_function(1024,tokenizer)
    data_files = {}
    data_files["train"] = '/home/zhangkechi/workspace/nlp_course/data/train.jsonl'
    raw_datasets = load_dataset("json", data_files=data_files)
    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.map(
                preprocess_function,
                with_indices = True,
                batched=True,
                remove_columns=train_dataset.column_names
            )
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn = data_collator)
    return dataloader

if __name__ == '__main__':
    dataloader = unit_test()
    for batch in dataloader:
        import pdb;pdb.set_trace()
    
    import pdb;pdb.set_trace()