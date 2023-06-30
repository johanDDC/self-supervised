import torch
import torchaudio.transforms as T


class LogMelSpec(T.MelSpectrogram):
    def __init__(self, min_eps, max_eps, **kwargs):
        super().__init__(**kwargs)
        self.min_eps = min_eps
        self.max_eps = max_eps

    def forward(self, waveform):
        return torch.log(
            (T.MelSpectrogram.forward(self, waveform) + self.min_eps)
            .clamp_(min=self.min_eps, max=self.max_eps)
        )
