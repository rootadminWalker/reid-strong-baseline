# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp

import cv2 as cv
from PIL import Image
from torch.utils.data import Dataset


def to_rgb(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def to_bgr(img):
    return img


def to_hsv(img):
    return cv.cvtColor(img, cv.COLOR_BGR2HSV)


color_func = {
    'rgb': to_rgb,
    'bgr': to_bgr,
    'hsv': to_hsv
}


def read_image(img_path, color_space):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            # img = Image.open(img_path).convert('RGB')
            img = cv.imread(img_path)
            # if use_rgb:
            #     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = color_func[color_space](img)

            img = Image.fromarray(img)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, color_space, transform=None):
        self.dataset = dataset
        self.color_space = color_space
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path, self.color_space)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path
