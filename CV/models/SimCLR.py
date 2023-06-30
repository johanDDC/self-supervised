import torch
import torch.nn as nn
import torchvision.models as mdls
from torch.utils.data import DataLoader
from tqdm import tqdm

import CV.configs.SimCLR
from CV.utils import ContrastiveLoss


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = mdls.resnet18(num_classes=10)
        self.encoder.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.fc(self.encode(x))


def pretrain(cfg: CV.configs.SimCLR.SimCLRConfig, pretrain_data, **kwargs):
    model = SimCLR()
    loader = DataLoader(pretrain_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=True,
                        num_workers=kwargs["num_workers"], drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    total_steps=cfg.num_epochs + 1, max_lr=3e-4, anneal_strategy="cos",
                                                    pct_start=cfg.scheduler_num_warmup_epochs / cfg.num_epochs)
    criterion = ContrastiveLoss(temp=cfg.loss_temperature)

    pbar = tqdm(range(1 + cfg.num_epochs))
    for epoch in pbar:
        losses = torch.zeros(1, dtype=torch.float32, device=kwargs["device"])
        for (imgs_a, imgs_b), _ in loader:
            imgs_a = imgs_a.to(kwargs["device"], non_blocking=True)
            imgs_b = imgs_b.to(kwargs["device"], non_blocking=True)

            repr_a = model(imgs_a)
            repr_b = model(imgs_b)

            loss = criterion(repr_a, repr_b)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses += loss.detach()
        losses /= len(loader)
        pbar.set_description(f"Epoch {epoch}, loss: {losses}")
        pbar.update(1)
        scheduler.step()

    return model.encoder
