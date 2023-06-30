import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration


class T5EncoderWithHead(nn.Module):
    def __init__(self, config: T5Config,
                head_out_features: int):
        super().__init__()
        config.feed_forward_proj = "gated_gelu"
        model = T5ForConditionalGeneration(config)
        self.encoder = model.encoder
        self.head = nn.Linear(config.d_model, head_out_features)
        del model

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x[0])