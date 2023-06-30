import torch

from torch.nn.utils.rnn import pad_sequence

from NLP.src.utils.collators.BaseCollator import BaseCollator


class MLMCollator(BaseCollator):
    def __init__(self, tokenizer):
        tokenizer.add_special_tokens({"mask_token": "<MASK>"})
        super().__init__(tokenizer)
        self.mask_id = tokenizer.mask_token_id

    def __call__(self, batch):
        input_ids = []
        for sample in batch:
            input_ids.append(torch.tensor(sample["input_ids"]))
        input_ids = pad_sequence(input_ids, padding_value=self.pad_id, batch_first=True)
        mask = (torch.rand(input_ids.shape) <= 0.15)
        for sp_tok in self.special_tokens:
            mask &= (input_ids != sp_tok)
        unmasked_tokens = mask & (torch.rand(input_ids.shape) <= 0.1)
        rand_mask = mask & (torch.rand(input_ids.shape) <= 0.1)
        rand_mask_ids = torch.nonzero(rand_mask, as_tuple=True)
        rand_tokens = torch.randint(4, self.tokenizer.vocab_size - 101,
                                    (len(rand_mask_ids[0]),))

        target = input_ids.clone()
        target.masked_fill_(~mask, self.pad_id)
        mask &= (~rand_mask & ~unmasked_tokens)
        input_ids.masked_fill_(mask, self.mask_id)
        input_ids[rand_mask_ids] = rand_tokens

        return input_ids, target
