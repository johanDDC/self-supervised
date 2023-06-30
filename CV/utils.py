from random import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Probing(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = deepcopy(encoder)
        self.fc = torch.nn.Linear(512, 10)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.fc(self.encode(x))


class ContrastiveLoss(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.T = temp
        self.mask = None
        self.labels = None

    def forward(self, repr1, repr2):
        representations = torch.cat([repr1, repr2], dim=0)
        representations = F.normalize(representations, dim=1)
        similarity_matrix = representations @ representations.T

        if self.mask is None:
            self.mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool, device=device)

        if self.labels is None:
            self.labels = torch.cat([torch.arange(repr1.shape[0]) for i in range(2)], dim=0).float().to(device)
            self.labels = (self.labels.unsqueeze(0) == self.labels.unsqueeze(1))
            self.labels = self.labels.masked_select(~self.mask).view(self.labels.shape[0], -1).bool()

        similarity_matrix = similarity_matrix.masked_select(~self.mask).view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix.masked_select(self.labels).view(similarity_matrix.shape[0], -1)
        negatives = similarity_matrix.masked_select(~self.labels).view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

        logits = logits / self.T
        loss = F.cross_entropy(logits, labels)
        return loss


def mixup_data(batch, targets, alpha, device):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = batch.size()[0]
    index = torch.randperm(batch_size)
    index.to(device)

    mixed_x = lam * batch + (1 - lam) * batch[index, :]
    y_a, y_b = targets, targets[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
