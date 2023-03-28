import sys

import numpy as np
import torchvision.transforms as T

sys.path.append('.')
from data import make_data_loaders_with_stages
from utils import setup_cli
import torch


def main(cfg):
    mean = torch.tensor([0., 0., 0.])
    std = torch.tensor([0., 0., 0.])
    c = 0

    transform = T.Compose([
        T.ToTensor(),
        T.Resize(cfg.INPUT.SIZE_TEST),
    ])
    train_loaders, val_loader, *_ = make_data_loaders_with_stages(cfg)
    train_loaders[0].dataset.transform = transform
    val_loader.dataset.transform = transform
    for images, *_ in train_loaders[0]:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        c += 1
    for images, *_ in val_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        c += 1

    mean /= len(train_loaders[0].dataset)
    std /= len(train_loaders[0].dataset)
    #
    # print(mean, std)
    # count = c * cfg.SOLVER.IMS_PER_BATCH * cfg.INPUT.SIZE_TRAIN[0] * cfg.INPUT.SIZE_TRAIN[1]
    # mean = mean / count
    # std = (std / count - mean ** 2).sqrt()

    print(mean, std)


if __name__ == '__main__':
    cfg, _ = setup_cli()
    main(cfg)
