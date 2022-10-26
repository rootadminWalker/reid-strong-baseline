# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from pytorch_metric_learning.losses import CentroidTripletLoss
from pytorch_metric_learning.reducers import DoNothingReducer

from .triplet_loss import TripletLoss, EuclideanDistance
from .id_loss import CrossEntropyLabelSmooth, AMSoftmaxLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif sampler == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif sampler == 'softmax_triplet':
        def loss_func(score, feat, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                loss_triplet = triplet(feat, target)[0]
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    # return xent(score, target) + triplet(feat, target)[0]
                    loss_classification = xent(score, target)
                else:
                    # return F.cross_entropy(score, target) + triplet(feat, target)[0]
                    loss_classification = F.cross_entropy(score, target)
                loss_total = loss_classification + loss_triplet
                return loss_total, loss_classification, loss_triplet
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


def make_loss_with_center(cfg, num_classes):    # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048

    assert cfg.MODEL.METRIC_LOSS_TYPE in ['center', 'triplet_center', 'am_triplet_center', 'am_CTL_triplet_center', "CTL_triplet_center"], '''
            expected METRIC_LOSS_TYPE with center should be center, triplet_center, am_triplet_center'
            'but got {}'''.format(cfg.MODEL.METRIC_LOSS_TYPE)

    print("Criterions: ")
    if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
        print(center_criterion)
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        print(triplet)
    if 'CTL' in cfg.MODEL.METRIC_LOSS_TYPE:
        distance = EuclideanDistance()
        ctl = CentroidTripletLoss(margin=cfg.SOLVER.MARGIN, distance=distance, reducer=DoNothingReducer())
        print(ctl)

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        if 'am' in cfg.MODEL.METRIC_LOSS_TYPE:
            xent = AMSoftmaxLoss(s=cfg.SOLVER.AM_S, m=cfg.SOLVER.AM_M, num_classes=num_classes, epsilon=cfg.SOLVER.ID_EPSILON)
        else:
            xent = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=cfg.SOLVER.ID_EPSILON)     # new add by luo
        print("label smooth on, ", end='')
    else:
        if 'am' in cfg.MODEL.METRIC_LOSS_TYPE:
            xent = AMSoftmaxLoss(num_classes=num_classes, epsilon=0)
        else:
            xent = F.cross_entropy
    print(xent)
    print("numclasses:", num_classes)

    def loss_func(score, feat, target):
        loss_center = cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
        loss_classification = xent(score, target)
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            loss_total = loss_classification + loss_center
            return loss_total, loss_classification, loss_center

        elif 'triplet_center' in cfg.MODEL.METRIC_LOSS_TYPE:
            loss_triplet = triplet(feat, target)[0]
            if 'CTL' in cfg.MODEL.METRIC_LOSS_TYPE:
                loss_ctl = ctl(feat, target)
                loss_total = loss_classification + loss_triplet + loss_center + loss_ctl
                return loss_total, loss_classification, loss_center, loss_triplet, loss_ctl
            else:
                loss_total = loss_classification + loss_triplet + loss_center
                return loss_total, loss_classification, loss_center, loss_triplet 

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    return loss_func, center_criterion