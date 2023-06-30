import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration


class T5DecoderWithHead(nn.Module):
    def __init__(self, config: T5Config,
                head_out_features: int):
        super().__init__()
        config.feed_forward_proj = "gated_gelu"
        model = T5ForConditionalGeneration(config)
        self.decoder = model.decoder
        self.head = nn.Linear(config.d_model, head_out_features, bias=False)
        del model

    def forward(self, x):
        x = self.decoder(x)
        return self.head(x[0])