import sys

import cv2 as cv
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from pytorch_lightning.callbacks import RichProgressBar

sys.path.append('.')
from data.datasets import init_dataset
from data import make_val_dataset
from engine.reid_module import PersonReidModule
from utils import setup_cli, setup_loggers


class OpenVINOModule(PersonReidModule):
    def __init__(self, cfg, train_num_classes, val_num_queries):
        super(OpenVINOModule, self).__init__(cfg, train_num_classes, val_num_queries)
        self.net = cv.dnn.readNet(cfg.MODEL.PRETRAIN_PATH.split('.')[0] + '.bin', cfg.MODEL.PRETRAIN_PATH)

    def validation_step(self, batch, batch_idx):
        data, pids, camids = batch
        data = data.detach().numpy()
        feats = []
        for img in data:
            blob = cv.dnn.blobFromImage(
                img,
                scalefactor=1.0,
                mean=(0, 0, 0),
                swapRB=False,
                crop=False
            )
            self.net.setInput(blob)
            feat = self.net.forward()
            feats.append(torch.tensor(feat, device='cuda:0'))

        feats = torch.cat(feats)
        self.metric.update([feats, pids, camids])

    def on_validation_epoch_end(self):
        super(OpenVINOModule, self).on_validation_epoch_end()


def get_image_label(image_name):
    return image_name.split('_')[0]


def main(cfg):
    val_dataset = init_dataset(
        cfg.DATASETS.VAL_NAMES,
        root=cfg.DATASETS.VAL_ROOT,
        aug_per_image=cfg.SOLVER.AUG_PER_IMG
    )
    _, val_loader, val_num_queries, val_num_classes, _ = make_val_dataset(cfg, val_dataset)
    val_loader.dataset.transform = T.Compose([
        T.Resize(size=cfg.INPUT.SIZE_TRAIN),
        T.Lambda(lambda x: torch.tensor(np.array(x)))
    ])

    model = OpenVINOModule(cfg, val_num_classes, val_num_queries)
    print(len(val_loader), val_num_queries, val_num_classes)
    trainer = pl.Trainer(
        accelerator='cpu',
        benchmark=True,
        num_sanity_val_steps=0,
        callbacks=[RichProgressBar()]
    )
    trainer.validate(model, dataloaders=val_loader)


if __name__ == '__main__':
    cfg, args = setup_cli()
    logger = setup_loggers(args)
    main(cfg)
