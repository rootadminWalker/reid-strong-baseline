import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import RichProgressBar

from data import make_pl_datamodule
from engine.reid_module import PersonReidModule
from utils import setup_cli, setup_loggers


def get_image_label(image_name):
    return image_name.split('_')[0]


def main(cfg):
    datamodule = make_pl_datamodule(cfg)

    model = PersonReidModule(cfg, datamodule.num_classes, datamodule.num_queries)
    # model = model.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH)
    # model.cuda()
    # model.to_onnx('/tmp/88_95.4_bgr.onnx', torch.randn((1, 3, *cfg.INPUT.SIZE_TRAIN)).cuda(), export_params=True)
    trainer = pl.Trainer(
        accelerator=cfg.MODEL.DEVICE,
        devices=list(map(int, cfg.MODEL.DEVICE_ID)),
        benchmark=True,
        num_sanity_val_steps=0,
        callbacks=[RichProgressBar()]
    )
    trainer.validate(model, datamodule=datamodule)


if __name__ == '__main__':
    cfg, args = setup_cli()
    logger = setup_loggers(args)
    main(cfg)
