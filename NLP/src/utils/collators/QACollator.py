import torch

from torch.nn.utils.rnn import pad_sequence

from NLP.src.utils.collators.BaseCollator import BaseCollator


class QACollator(BaseCollator):
    def __call__(self, batch):
        input_ids = []
        start_positions = []
        end_positions = []
        for sample in batch:
            input_ids.append(torch.tensor(sample["input_ids"], dtype=torch.long))
            start_positions.append(sample["start_positions"])
            end_positions.append(sample["start_positions"])

        batch = {
            "input_ids": pad_sequence(input_ids, padding_value=self.pad_id, batch_first=True),
            "start_positions": torch.tensor(start_positions, dtype=torch.long),
            "end_positions": torch.tensor(start_positions, dtype=torch.long)
        }
        batch["attention_mask"] = (batch["input_ids"] != self.pad_id).clone()

        return batch
