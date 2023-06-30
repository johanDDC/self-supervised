import os

import torchvision.transforms as transforms
import torchvision.datasets as dsets


class Augmentator():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        return self.transforms(x), self.transforms(x)


test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(15,),
    transforms.RandomCrop(96),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    ),
])

pretrain_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(size=96),
    transforms.RandomApply([
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.GaussianBlur(96 // 10, sigma=(0.1, 2.0)),
])


def get_train_data(root_dir, train_transforms=train_transforms,
                   test_transforms=test_transforms):
    stl10_train = dsets.STL10(root=os.path.join(root_dir, "train"),
                              split="train", download=True, transform=train_transforms)
    stl10_test = dsets.STL10(root=os.path.join(root_dir, "test"),
                             split="test", download=True, transform=test_transforms)
    return stl10_train, stl10_test


def get_pretrain_data(root_dir):
    stl10_unlabeled = dsets.STL10(root=os.path.join(root_dir, "unlabeled"), split="unlabeled",
                                  transform=Augmentator(pretrain_transforms), download=True)
    return stl10_unlabeled
