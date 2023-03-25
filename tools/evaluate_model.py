import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar

sys.path.append('.')
from data import make_val_dataset
from engine.reid_module import PersonReidModule
from utils import setup_cli, setup_loggers


def get_image_label(image_name):
    return image_name.split('_')[0]


def main(cfg):
    _, val_loader, val_num_queries, val_num_classes = make_val_dataset(cfg)

    model = PersonReidModule(cfg, val_num_classes, val_num_queries)
    model = model.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH)
    print(len(val_loader), val_num_queries, val_num_classes)
    # model.cuda()
    # model.to_onnx('/tmp/88_95.4_bgr.onnx', torch.randn((1, 3, *cfg.INPUT.SIZE_TRAIN)).cuda(), export_params=True)
    trainer = pl.Trainer(
        accelerator=cfg.MODEL.DEVICE,
        devices=list(map(int, cfg.MODEL.DEVICE_ID)),
        benchmark=True,
        num_sanity_val_steps=0,
        callbacks=[RichProgressBar()]
    )
    trainer.validate(model, dataloaders=val_loader)


if __name__ == '__main__':
    cfg, args = setup_cli()
    logger = setup_loggers(args)
    main(cfg)
