import torch

from torch.nn.utils.rnn import pad_sequence

from NLP.src.utils.collators.BaseCollator import BaseCollator


class MASSCollator(BaseCollator):
    def __init__(self, tokenizer, one_seq=True):
        tokenizer.add_special_tokens({"mask_token": "<MASK>"})
        super().__init__(tokenizer)
        self.mask_id = tokenizer.mask_token_id
        self.one_seq = one_seq

    def __call__(self, batch):
        input_ids = []
        input_lens = []
        for sample in batch:
            input_ids.append(torch.tensor(sample["input_ids"]))
            input_lens.append(len(sample["input_ids"]))
        input_ids = pad_sequence(input_ids, padding_value=self.pad_id, batch_first=True)
        input_lens = torch.tensor(input_lens)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        decoder_mask = torch.zeros((input_ids.shape[0],
                                    input_ids.shape[1] + 1), dtype=torch.bool)
        for i in range(len(input_lens)):
            if self.one_seq:
                k = input_lens[i] // 2
                margin = torch.randint(0, input_lens[i] - k, (1,))
                mask[i][margin:margin + k] = True
                decoder_mask[i][margin + k] = True
            else:
                k = input_lens[i] // 4
                margin = [torch.randint(0, input_lens[i] - k, (1,))]
                margin.append(
                    torch.randint(margin[0], input_lens[i] - k, (1,)),
                )
                mask[i][margin[0]:margin[0] + k] = True
                mask[i][margin[1]:margin[1] + k] = True
                decoder_mask[i][margin[0] + k - 1] = True
                decoder_mask[i][margin[1] + k - 1] = True

        batch = {
            "encoder_input": input_ids.masked_fill(mask, self.mask_id),
            "decoder_input": torch.hstack([
                torch.full((input_ids.shape[0], 1), self.mask_id),
                input_ids.masked_fill(~mask, self.mask_id)
            ]).masked_fill(decoder_mask, self.mask_id),
            "target": input_ids.masked_fill(~mask, self.mask_id)
        }

        return batch
