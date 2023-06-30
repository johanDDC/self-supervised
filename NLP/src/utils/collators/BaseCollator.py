import torch

from typing import List

from transformers import T5TokenizerFast

from torch.nn.utils.rnn import pad_sequence


class BaseCollator:
    def __init__(self, tokenizer: T5TokenizerFast, add_bos_eos=False):
        self.tokenizer = tokenizer
        self.pad_id: int = tokenizer.pad_token_id

        self.special_tokens: List[int] = tokenizer.all_special_ids
        self.__add_bos_eos = add_bos_eos

    def __call__(self, batch):
        input_ids = []
        labels = []
        for sample in batch:
            input_ids.append(torch.tensor(sample["input_ids"], dtype=torch.long))
            labels.append(torch.tensor(sample["labels"], dtype=torch.long))

        batch = {
            "input_ids": pad_sequence(input_ids, padding_value=self.pad_id, batch_first=True),
            "labels": pad_sequence(labels, padding_value=-100, batch_first=True)
        }
        batch["attention_mask"] = (batch["input_ids"] != self.pad_id).clone()

        return batch


class BaseTestCollator(BaseCollator):
    def __call__(self, batch):
        batch = BaseCollator.__call__(self, batch)
        targets = batch["labels"]
        del batch["labels"]
        return batch, targets
