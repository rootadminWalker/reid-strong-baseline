# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from .backbones.resnet import resnet50, resnet101
from .backbones.resnet_ibn import resnet34_ibn_a, resnet50_ibn_a, resnet101_ibn_a
from .backbones.resnext_ibn import resnext101_ibn_a

MODELS = {
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet34_ibn_a": resnet34_ibn_a,
    "resnet50_ibn_a": resnet50_ibn_a,
    "resnet101_ibn_a": resnet101_ibn_a,
    "resnext101_ibn_a": resnext101_ibn_a
}


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, neck, neck_feat, model_name, pretrain_choice,
                 norm_classifier_w=False):
        super(Baseline, self).__init__()
        self.is_pretrain = pretrain_choice == 'imagenet'
        self.base = self.__load_base(model_name, last_stride, self.is_pretrain)
        if self.is_pretrain:
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.norm_classifier_w = norm_classifier_w

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    @staticmethod
    def __load_base(model_name, last_stride, pretrained):
        assert model_name in MODELS, "This model is currently unavailable"
        return MODELS[model_name](last_stride, pretrained=pretrained)

    def base_forward(self, x):
        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)

        feat = global_feat
        if self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)
        return feat, global_feat

    def train_forward(self, x):
        feat, global_feat = self.__base_forward(x)
        if self.norm_classifier_w:
            feat = F.normalize(feat, p=2, dim=1)
            with torch.no_grad():
                self.classifier.weight.div_(torch.norm(self.classifier.weight, dim=1, keepdim=True))
        cls_score = self.classifier(feat)
        return cls_score, global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model']
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
