# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import argparse
import os

from config import cfg
from .logger import _setup_loggers


def setup_cli():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    output_dir = cfg.OUTPUT_DIR
    tb_logs_path = os.path.join(output_dir, 'tb_logs')
    cfg.TB_LOG_DIR = tb_logs_path
    cfg.freeze()

    if cfg.MODE == "train":
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.makedirs(tb_logs_path)

    return cfg, args


def setup_loggers(args, output_dir=None):
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    logger = _setup_loggers("reid_training", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    return logger

