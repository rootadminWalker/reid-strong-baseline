# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from utils import weights_init_kaiming
from .backbones.osnet import osnet_ibn_x1_0
from .backbones.osnet import osnet_x1_0
from .backbones.osnet_ain import osnet_ain_x1_0
from .backbones.resnet import resnet50, resnet101
from .backbones.resnet_ibn import resnet34_ibn_a, resnet50_ibn_a, resnet101_ibn_a
from .backbones.resnext_ibn import resnext101_ibn_a
from layers import GeneralizedMeanPooling

MODELS = {
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet34_ibn_a": resnet34_ibn_a,
    "resnet50_ibn_a": resnet50_ibn_a,
    "resnet101_ibn_a": resnet101_ibn_a,
    "resnext101_ibn_a": resnext101_ibn_a,
    'osnet_x1_0': osnet_x1_0,
    'osnet_ibn_x1_0': osnet_ibn_x1_0,
    'osnet_ain_x1_0': osnet_ain_x1_0,
}


class Baseline(nn.Module):
    # in_planes = 2048

    def __init__(self, num_classes, last_stride, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        self.is_pretrain = pretrain_choice == 'imagenet'
        if self.is_pretrain:
            print('Loading pretrained ImageNet model......')
        self.base = self.__load_backbone(model_name, last_stride, self.is_pretrain)

        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.in_planes = self.__get_in_planes(model_name)
        self.pooling = nn.AdaptiveAvgPool2d(1)

        if self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.bottleneck.apply(weights_init_kaiming)
        elif self.neck == 'GeM':
            self.pooling = GeneralizedMeanPooling(p=3, eps=1e-6)
            linear = nn.Linear(in_features=self.in_planes, out_features=self.in_planes)
            bn = nn.BatchNorm1d(self.in_planes)
            bn.bias.requires_grad_(False)  # no shift
            PReLU = nn.PReLU()

            bn.apply(weights_init_kaiming)
            linear.apply(weights_init_kaiming)

            self.bottleneck = nn.Sequential(linear, bn, PReLU)

    @staticmethod
    def __get_in_planes(model_name):
        if model_name == 'resnet18' or model_name == 'resnet34' or 'osnet' in model_name:
            return 512
        else:
            return 2048

    @staticmethod
    def __show_available_models():
        msg = 'Available models:\n'
        for model_name in MODELS.keys():
            msg += f'{model_name}\n'
        return msg

    def __load_backbone(self, model_name, last_stride, pretrained):
        assert model_name in MODELS, f"This model is currently unavailable\n{self.__show_available_models()}"
        return MODELS[model_name](last_stride, pretrained=pretrained)

    def base_forward(self, x):
        global_feat = self.pooling(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)

        out_feat = global_feat
        if self.neck == 'bnneck':
            out_feat = self.bottleneck(global_feat)

        return out_feat, global_feat

    def train_forward(self, x):
        out_feat, global_feat = self.base_forward(x)
        return out_feat, global_feat

    def forward(self, x):
        out_feat, global_feat = self.base_forward(x)
        if self.neck_feat == 'after':
            return out_feat
        else:
            return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model']
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
