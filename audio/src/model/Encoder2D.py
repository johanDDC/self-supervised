import torch
import torch.nn as nn

from torchvision.models import resnet18


# class Encoder2D(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.in_conv = nn.Conv2d(1, 16, kernel_size=7, stride=2)
#         self.block1 = nn.Sequential(
#             nn.Conv2d(16, 16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.LeakyReLU(.2, inplace=True),
#             nn.Conv2d(16, 16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16),
#         )
#         self.res_conv1 = nn.Conv2d(16, 16, kernel_size=1)
#         self.downsample = nn.Conv2d(16, 32, kernel_size=2, stride=2)
#         self.block2 = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(.2, inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#         )
#         self.res_conv2 = nn.Conv2d(32, 32, kernel_size=1)
#
#     def forward(self, x):
#         x = self.in_conv(x)
#         x = self.block1(x) + self.res_conv1(x)
#         x = self.downsample(x)
#         x = self.block2(x) + self.res_conv2(x)
#         return x

class Encoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18()
        self.model.fc = nn.Identity()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.model(x)