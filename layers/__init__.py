# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth, AdMSoftmaxLabelsSmooth
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

    assert cfg.MODEL.METRIC_LOSS_TYPE in ['center', 'triplet_center', 'am_triplet_center'], '''
            expected METRIC_LOSS_TYPE with center should be center, triplet_center, am_triplet_center'
            'but got {}'''.format(cfg.MODEL.METRIC_LOSS_TYPE)

    # center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
    # elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        # center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        if cfg.MODEL.METRIC_LOSS_TYPE == 'am_triplet_center':
            xent = AdMSoftmaxLabelsSmooth(num_classes=num_classes)
        else:
            xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, ", end='')
    else:
        if cfg.MODEL.METRIC_LOSS_TYPE == 'am_triplet_center':
            xent = AdMSoftmaxLabelsSmooth(num_classes=num_classes, epsilon=0)
        else:
            xent = F.cross_entropy
    print("numclasses:", num_classes)

    def loss_func(score, feat, target):
        loss_center = cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            # if cfg.MODEL.IF_LABELSMOOTH == 'on':
            #     loss_classification = xent(score, target)
            # else:
            #     loss_classification = F.cross_entropy(score, target)
            loss_classification = xent(score, target)
            loss_total = loss_classification + loss_center
            return loss_total, loss_classification, loss_center

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            # if cfg.MODEL.IF_LABELSMOOTH == 'on':
            #     loss_classification = xent(score, target)
            # else:
            #     loss_classification = F.cross_entropy(score, target)
            loss_classification = xent(score, target)
            loss_triplet = triplet(feat, target)[0]
            loss_total = loss_classification + loss_triplet + loss_center
            return loss_total, loss_classification, loss_center, loss_triplet 

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    return loss_func, center_criterion