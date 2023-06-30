import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def crop_collator(batch, ratio=0.3, overlap=False):
    speakers = []
    wav_crops = []
    mel_crops = []
    for i in range(len(batch)):
        speakers.append(batch[i]["speaker"])
        wav_len = batch[i]["wav"].shape[1]
        crop_len = int(ratio * wav_len)

        gap = 0
        margin_left = 0
        if overlap is not None:
            if overlap:
                min_gap = 0 if ratio <= 0.5 else int((ratio - (1 - ratio)) * wav_len)
                gap = -np.random.randint(min_gap, crop_len)
            else:
                gap = np.random.randint(0, wav_len - 2 * crop_len)
            max_margin = wav_len - 2 * crop_len - gap
            if max_margin > 0:
                margin_left = np.random.randint(0, max_margin)
        coin = torch.bernoulli(torch.tensor(0.5))

        if coin == 1:
            wav_crops.append(batch[i]["wav"][0, margin_left:margin_left + crop_len])
            mel_crops.append(batch[i]["wav"][0, margin_left + crop_len + gap:margin_left + 2 * crop_len + gap])
        else:
            mel_crops.append(batch[i]["wav"][0, margin_left:margin_left + crop_len])
            wav_crops.append(batch[i]["wav"][0, margin_left + crop_len + gap:margin_left + 2 * crop_len + gap])

    return {
        "speaker": torch.tensor(speakers),
        "wav_crops": pad_sequence(wav_crops, batch_first=True).unsqueeze(1),
        "mel_crops": pad_sequence(mel_crops, batch_first=True).unsqueeze(1),
    }
