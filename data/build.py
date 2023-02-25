# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler  # New add by gu
from .transforms import build_transforms


def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, aug_per_image=cfg.SOLVER.AUG_PER_IMG)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, aug_per_image=cfg.SOLVER.AUG_PER_IMG)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, transform=train_transforms)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes


def make_data_loaders_with_stages(cfg):
    assert len(cfg.SOLVER.STAGE_TRANSFORMS) == len(cfg.SOLVER.STAGE_PERIOD), \
        f"STAGE_PERIOD and STATE_TRANSFORMS must have the same length, " \
        f"but got {cfg.SOLVER.STAGE_TRANSFORMS} and {cfg.SOLVER.STAGE_PERIOD} "
    train_loaders = []

    use_rgb = cfg.DATALOADER.USE_RGB
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, aug_per_image=cfg.SOLVER.AUG_PER_IMG)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    num_classes = dataset.num_train_pids

    val_transforms = build_transforms_stage(cfg, is_train=False)
    val_set = ImageDataset(dataset.query + dataset.gallery, use_rgb, transform=val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    for st_idx in range(len(cfg.SOLVER.STAGE_TRANSFORMS) + 1):
        stage_transforms = cfg.SOLVER.STAGE_TRANSFORMS[:st_idx]
        train_transforms = build_transforms_stage(cfg, stage_transforms=stage_transforms, is_train=True)
        train_set = ImageDataset(dataset.train, use_rgb, transform=train_transforms)
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
