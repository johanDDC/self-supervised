import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

        self.__criterion = nn.CrossEntropyLoss()
        self.__similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, x):
        N = x.shape[0] // 2
        similarities = self.__similarity(x.unsqueeze(0), x.unsqueeze(1))
        logits = similarities.flatten()[1:].view(2 * N - 1, 2 * N + 1)[:, :-1].reshape(2 * N, 2 * N - 1)
        labels = torch.cat([torch.arange(N - 1, 2 * N - 1), torch.arange(N)]).to(x.device, non_blocking=True)
        return self.__criterion(logits / self.temperature, labels)
