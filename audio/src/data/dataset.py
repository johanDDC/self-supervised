import torch
import torchaudio

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class AudioDataset(Dataset):
    def __init__(self, df):
        self.target = df["y"].values.astype(int)
        self.speakers = df["speaker"].values.astype(int)
        self.wavs = self.__load_wavs(df["paths"])

    def __load_wavs(self, paths):
        wavs = []
        for path in paths:
            waveform = torchaudio.load(path, normalize=True)[0]
            wavs.append(waveform)
        return wavs

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return {
            "speaker": self.speakers[idx],
            "target": self.target[idx],
            "wav": self.wavs[idx],
        }

    @staticmethod
    def collator(batch):
        speakers = []
        targets = []
        wavs = []
        for i in range(len(batch)):
            speakers.append(batch[i]["speaker"])
            targets.append(batch[i]["target"])
            wavs.append(batch[i]["wav"].squeeze())

        return {
            "speaker": torch.tensor(speakers),
            "target": torch.tensor(targets),
            "wav": pad_sequence(wavs, batch_first=True).unsqueeze(1),
        }