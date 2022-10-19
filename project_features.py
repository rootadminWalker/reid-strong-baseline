import os

import matplotlib.pyplot as plt
from config import cfg
from modeling import build_model
from utils.logger import setup_logger
from data.transforms import build_transforms
from torch.backends import cudnn
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
import cv2 as cv
from torchinfo import summary
import argparse
from sklearn.decomposition import PCA


def get_label_image(image_name):
    return image_name.split('_')[0]


def main(args):
    device = f'cuda:{args.device_id}'

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TEST.WEIGHT = args.checkpoint_path
    cfg.MODEL.PRETRAIN_PATH = 'pretrained_weights/r50_ibn_a.pth'
    cfg.MODEL.NAME = 'resnet50_ibn_a'
    cfg.freeze()

    logger = setup_logger("reid_baseline", '/tmp', 0)
    # logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    model = build_model(cfg, 751)
    model.load_param(cfg.TEST.WEIGHT)
    model.eval()
    model.to(device)
    summary(model, input_size=(1, 3, *cfg.INPUT.SIZE_TRAIN))

    transforms = build_transforms(cfg, is_train=False)

    dataset_path = os.path.join(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.NAMES, 'bounding_box_train')
    pca = PCA(n_components=2)


    color_map = {
        0: 'tab:red',
        1: 'tab:blue',
        2: 'tab:gray',
        3: 'tab:purple',
        4: 'tab:orange',
    }

    plt.style.use("ggplot")
    plt.figure()
    # plt.xlim(-6.2, 6.5)
    # plt.ylim(-5, 6.9)
    i = 0
    features = []
    d = os.listdir(dataset_path)
    d.remove('Thumbs.db')
    for idx, image_name in enumerate(d):
        label = get_label_image(image_name)

        image_path = os.path.join(dataset_path, image_name)
        image = cv.imread(image_path)
        blob = transforms(image).unsqueeze(0).to(device)
        descriptor = model(blob).detach().squeeze(0).cpu().numpy()
        features.append(descriptor)

        if idx == int(len(d) / len(color_map)):
            i += 1
            features = np.array(features)
            print(features.shape)
            features = pca.fit_transform(features)
            plt.scatter(features[:, 0], features[:, 1], color=color_map[i])
            features = []
        
    # epoch = t + 1
    plt.title(f"Feature distribution baseline")
    plt.xlabel("Embedding dim 1")
    plt.ylabel(f"Embedding dim 2")
    # plt.savefig(os.path.join(cluster_plots_path, f'ep{epoch :>03d}'))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )


    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--selected-labels', nargs='+', type=str)
    parser.add_argument('--image-per-label', type=int, required=True)
    parser.add_argument('--device-id', type=int)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    main(args)
