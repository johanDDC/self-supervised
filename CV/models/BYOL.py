import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as mdls
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import CV.configs.SimCLR


class TargetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = mdls.resnet18(num_classes=10)
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )

    def encode(self, x):
        return self.encoder(x)

    def project(self, x):
        return self.projector(self.encode(x))

    def forward(self, x):
        return self.projector(self.encode(x))


class OnlineModel(TargetModel):
    def __init__(self):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        return self.predictor(self.project(x))


class BYOL(nn.Module):
    def __init__(self):
        super().__init__()
        self.online_model = OnlineModel()
        self.target_model = TargetModel()

        self.__init_weights()

    def __init_weights(self):
        for p in self.target_model.parameters():
            p.requires_grad_(False)

    def encode(self, x):
        return self.online_model.encode(x)

    def forward(self, imgs_a, imgs_b):
        prediction_a = self.online_model(imgs_a)
        prediction_b = self.online_model(imgs_b)

        with torch.no_grad():
            targets_a = self.target_model(imgs_b)
            targets_b = self.target_model(imgs_a)
        return prediction_a, prediction_b, targets_a, targets_b


class BYOL_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = F.normalize(x, dim=1, p=2)
        y = F.normalize(y, dim=1, p=2)
        return (2 - 2 * (x * y).sum(dim=-1)).mean()


def pretrain(cfg: CV.configs.BYOL.BYOLConfig, pretrain_data, **kwargs):
    model = BYOL()
    loader = DataLoader(pretrain_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=True,
                        num_workers=kwargs["num_workers"], drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    total_steps=cfg.num_epochs + 1, max_lr=3e-4, anneal_strategy="cos",
                                                    pct_start=cfg.scheduler_num_warmup_epoches / cfg.num_epochs)
    criterion = BYOL_loss()
    EMA = cfg.EMA

    pbar = tqdm(range(1 + cfg.num_epochs))
    for epoch in pbar:
        losses = torch.zeros(1, dtype=torch.float32, device=kwargs["device"])
        z_std = torch.zeros(1, dtype=torch.float32, device=kwargs["device"])
        for (imgs_a, imgs_b), _ in loader:
            imgs_a = imgs_a.to(kwargs["device"], non_blocking=True)
            imgs_b = imgs_b.to(kwargs["device"], non_blocking=True)

            prediction_a, prediction_b, targets_a, targets_b = model(
                imgs_a, imgs_b
            )

            loss = criterion(prediction_a, targets_a) + criterion(prediction_b, targets_b)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            for p, q in zip(model.target_model.parameters(),
                            model.online_model.parameters()):
                p.data = p.data * EMA + q.data * (1 - EMA)

            losses += loss.detach()
            with torch.no_grad():
                z = model.online_model.project(imgs_a)
                z_std += torch.trace(torch.cov(z)).detach()
        losses /= len(loader)
        z_std /= len(loader)
        pbar.set_description(f"Epoch {epoch}, loss: {losses}, z_std: {z_std}")
        pbar.update(1)
        scheduler.step()
        EMA = 1 - (1 - EMA) * (np.cos(np.pi * epoch / cfg.num_epochs) + 1) / 2

    return model.online_model.encoder
