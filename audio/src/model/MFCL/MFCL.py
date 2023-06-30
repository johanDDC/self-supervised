import torch
import torch.nn as nn

from src.model.Encoder1D import Encoder1D
from src.model.Encoder2D import Encoder2D
from src.utils.LogMelSpec import LogMelSpec


class MFCL(nn.Module):
    def __init__(self, transforms, device, **mel_kwargs):
        super().__init__()
        self.wav_encoder = Encoder1D()
        self.spec_encoder = Encoder2D()
        self.wav_transforms, self.mel_transforms = transforms
        self.wav_transforms, self.mel_transforms = self.wav_transforms.to(device), \
        self.mel_transforms.to(device)

        self.__spectrogram = LogMelSpec(**mel_kwargs).to(device)

    def forward(self, wav_crop, mel_crop=None):
        if mel_crop is None:
            mel_crop = wav_crop
        wav_crop = self.wav_transforms(wav_crop)
        wav_features = self.wav_encoder(wav_crop)

        mel_spec = self.__spectrogram(mel_crop)
        mel_spec = self.mel_transforms(mel_spec)
        spec_features = self.spec_encoder(mel_spec)

        return wav_features, spec_features
