# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler  # New add by gu
from .transforms import build_transforms_stage


class REIDDataModule(pl.LightningDataModule):
    def __init__(self, train_dataloaders, val_dataloader, num_queries, num_classes, stage_period):
        super(REIDDataModule, self).__init__()
        self._train_dataloaders = train_dataloaders
        self._val_dataloader = val_dataloader
        self._num_queries = num_queries
        self._num_classes = num_classes
        self._stage_period = stage_period
        self._stage_idx = 0

    @property
    def num_queries(self):
        return self._num_queries

    @property
    def num_classes(self):
        return self._num_classes

    def train_dataloader(self):
        if self._stage_idx < len(self._stage_period):
            if self.trainer.current_epoch == self._stage_period[self._stage_idx]:
                self._stage_idx += 1
        train_dataloader = self._train_dataloaders[self._stage_idx]
        return train_dataloader

    def val_dataloader(self):
        return self._val_dataloader


def make_data_loaders_with_stages(cfg):
    assert len(cfg.SOLVER.STAGE_TRANSFORMS) == len(cfg.SOLVER.STAGE_PERIOD), \
        f"STAGE_PERIOD and STATE_TRANSFORMS must have the same length, " \
        f"but got {cfg.SOLVER.STAGE_TRANSFORMS} and {cfg.SOLVER.STAGE_PERIOD} "
    train_loaders = []

    color_space = cfg.DATALOADER.COLOR_SPACE
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, aug_per_image=cfg.SOLVER.AUG_PER_IMG)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    num_classes = dataset.num_train_pids

    val_transforms = build_transforms_stage(cfg, is_train=False)
    val_set = ImageDataset(dataset.query + dataset.gallery, color_space, transform=val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    for st_idx in range(len(cfg.SOLVER.STAGE_TRANSFORMS) + 1):
        stage_transforms = cfg.SOLVER.STAGE_TRANSFORMS[:st_idx]
        train_transforms = build_transforms_stage(cfg, stage_transforms=stage_transforms, is_train=True)
        train_set = ImageDataset(dataset.train, color_space, transform=train_transforms)
        if cfg.DATALOADER.SAMPLER == 'softmax':
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=train_collate_fn
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE,
                                              fixed_epoch_steps=cfg.SOLVER.EVAL_INTERVAL),
                num_workers=num_workers, collate_fn=train_collate_fn
            )

        train_loaders.append(train_loader)

    return train_loaders, val_loader, len(dataset.query), num_classes


def make_pl_datamodule(cfg):
    return REIDDataModule(*make_data_loaders_with_stages(cfg), cfg.SOLVER.STAGE_PERIOD)
