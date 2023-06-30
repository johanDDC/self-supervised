import random
import datasets
import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
from copy import copy
from tqdm import tqdm


class FixedTokensSampler(Sampler):
    def __init__(
            self,
            dataset: datasets.arrow_dataset.Dataset,
            n_tokens: int,
            lengths: np.array = None,
            shuffle: bool = True
    ):
        self.n_tokens = n_tokens
        self.shuffle = shuffle

        if lengths is not None:
            self.lengths = lengths
        else:
            self.lengths = np.array([
                (i, len(sample)) for i, sample in enumerate(tqdm(dataset._data['input_ids']))
            ])

    def __iter__(self):
        if self.shuffle:
            self.lengths = self.lengths[np.random.permutation(len(self.lengths))]

        mean_length = np.mean(self.lengths, axis=0)[1]
        step = int(self.n_tokens / mean_length * 100)
        for i in range(0, len(self.lengths), step):
            pooled = sorted(self.lengths[i:i + step], key=lambda x: x[1])

            batches = []
            batch = []
            cur_n_tokens = 0
            for idx, length in pooled:
                if (len(batch) + 1) * length > self.n_tokens:
                    if len(batch) == 0:
                        print(f'Maximum number of tokens {self.n_tokens} is lower, than size of a sample {length}')
                        break
                    else:
                        batches.append(copy(batch))
                        batch = []
                        cur_n_tokens = 0
                else:
                    batch.append(idx)
                    cur_n_tokens += length

            if self.shuffle:
                random.shuffle(batches)

            for batch in batches:
                yield batch

    def __len__(self):
        return np.sum(self.lengths, axis=0)[1] // self.n_tokens


def collate_batch(batch, pad_id=0):
    input_ids = []
    for sample in batch:
        input_ids.append(torch.tensor(sample['input_ids']))

    return pad_sequence(input_ids, padding_value=pad_id, batch_first=True)
