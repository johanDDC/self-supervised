import torch.nn as nn


class ModelTransforms:
    def __init__(self, transforms, device):
        super().__init__()
        self.transforms = nn.Sequential(transforms)
        self.transforms.to(device)

    def __call__(self, batch):
        return self.transforms(batch)

