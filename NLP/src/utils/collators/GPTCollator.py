import torch

from torch.nn.utils.rnn import pad_sequence

from NLP.src.utils.collators.BaseCollator import BaseCollator


class MLMCollator(BaseCollator):
    def __init__(self, tokenizer):
        tokenizer.add_special_tokens({
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
        })
        super().__init__(tokenizer)
        self.bos = tokenizer.bos_token_id
        self.eos = tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id

    def __call__(self, batch):
        input_ids = []
        for sample in batch:
            input_ids.append(
                torch.tensor(sample['input_ids'] + [self.eos])
            )

        bos_col = torch.full((batch.shape[0], 1), self.bos)
        input_ids = pad_sequence(input_ids, padding_value=self.pad_id, batch_first=True)
        input_ids = torch.hstack([bos_col, input_ids])

        return input_ids
