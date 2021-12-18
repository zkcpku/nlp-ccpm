import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
from transformers.models.roberta.modeling_roberta import RobertaForMultipleChoice
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from torch.nn import CrossEntropyLoss


class BinaryRobertaChoice(RobertaForMultipleChoice):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.loss_weight = torch.tensor([0.8, 1.0])
        self.cross_entropy_loss = CrossEntropyLoss(weight = self.loss_weight)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # import pdb;pdb.set_trace()
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)


        loss = None
        if labels is not None:
            loss_fct = self.cross_entropy_loss
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {'loss': loss, 'logits':logits, 'hidden_states':outputs.hidden_states, 'attentions':outputs.attentions}


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
    from dataload.binary_dataset import unit_test
    dataloader = unit_test()
    

    config = AutoConfig.from_pretrained("ethanyt/guwenbert-base")
    model = BinaryRobertaChoice.from_pretrained(
        "ethanyt/guwenbert-base",
        config=config)

    def change_roberta_to_bert(model):
        try:
            o_embedding = model.roberta.embeddings.token_type_embeddings
            n_embedding = torch.nn.Embedding(2,768)
            n_embedding.weight = torch.nn.Parameter(o_embedding.weight.repeat(2,1))
            model.roberta.embeddings.token_type_embeddings = n_embedding
        except Exception as e:
            print(e)
        return model
    model = change_roberta_to_bert(model)

    for batch in dataloader:
        print([(k,batch[k].shape) for k in batch])
        output = model(**batch)
        import pdb;pdb.set_trace()

    