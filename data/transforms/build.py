# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing


def build_transforms(cfg, is_train=True):
    extra_augs = []
    if cfg.SOLVER.USE_AUGMIX:
        extra_augs.append(T.AugMix())
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            *extra_augs,
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
        print(f'Transforms: {transform}')
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Resize(cfg.INPUT.SIZE_TEST),
            normalize_transform
        ])

    return transform
