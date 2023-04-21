# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import warnings

from pytorch_metric_learning.losses import CentroidTripletLoss
from pytorch_metric_learning.reducers import MeanReducer

from .GeM import GeneralizedMeanPooling
from .center_loss import CenterLoss
from .id_loss import CrossEntropyHead, AMSoftmaxLoss, ArcFace, CurricularFace, SubCenterArcFace, \
    SubCenterCurricularFace, \
    CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss, EuclideanDistance

d_l = {'am': 0, 'arcface': 1, 'sub-center-arcface': 2,
       'curricularface': 3, 'sub-center-curricularface': 4, 'CTL': 5, 'triplet': 6, 'center': 7}


def center_loss(cfg, num_classes, feat_dim):
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    print(center_criterion)
    return center_criterion


def triplet_loss(cfg, num_classes, feat_dim):
    triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    print(triplet)
    return triplet


def CTL(cfg, num_classes, feat_dim):
    distance = EuclideanDistance()
    ctl = CentroidTripletLoss(margin=cfg.SOLVER.MARGIN, distance=distance, reducer=MeanReducer())
    print(ctl)
    return ctl


def id_loss(loss_sequences, cfg, num_classes, feat_dim):
    _biasON = cfg.MODEL.NECK != 'bnneck'
    xent = CrossEntropyLabelSmooth(num_classes=num_classes,
                                   epsilon=cfg.SOLVER.ID_EPSILON)
    if 'am' in loss_sequences:
        classification = AMSoftmaxLoss(
            in_features=feat_dim,
            num_classes=num_classes,
            s=cfg.SOLVER.AM_S,
            m=cfg.SOLVER.AM_M,
            epsilon=cfg.SOLVER.ID_EPSILON
        )
    elif 'arcface' in loss_sequences:
        # warnings.warn(f"Loss ArcFace does not support label smooth", UserWarning)
        classification = ArcFace(
            in_features=feat_dim,
            out_features=num_classes,
            s=cfg.SOLVER.AM_S,
            m=cfg.SOLVER.AM_M,
            epsilon=cfg.SOLVER.ID_EPSILON
        )
        classification.cross_entropy = xent
    elif 'sub-center-arcface' in loss_sequences:
        # warnings.warn(f"Loss Sub-center ArcFace does not support label smooth", UserWarning)
        classification = SubCenterArcFace(
            in_features=feat_dim,
            out_features=num_classes,
            s=cfg.SOLVER.AM_S,
            m=cfg.SOLVER.AM_M,
            epsilon=cfg.SOLVER.ID_EPSILON,
            K=cfg.SOLVER.AM_SUB_CENTERS
        )
    elif 'curricularface' in loss_sequences:
        # warnings.warn(f"Loss CurricularFace does not support label smooth", UserWarning)
        classification = CurricularFace(
            in_features=feat_dim,
            out_features=num_classes,
            s=cfg.SOLVER.AM_S,
            m=cfg.SOLVER.AM_M,
            epsilon=cfg.SOLVER.ID_EPSILON
        )
    elif 'sub-center-curricularface' in loss_sequences:
        classification = SubCenterCurricularFace(
            in_features=feat_dim,
            out_features=num_classes,
            s=cfg.SOLVER.AM_S,
            m=cfg.SOLVER.AM_M,
            epsilon=cfg.SOLVER.ID_EPSILON,
            K=cfg.SOLVER.AM_SUB_CENTERS
        )
    else:
        classification = CrossEntropyHead(in_features=feat_dim, num_classes=num_classes,
                                          epsilon=cfg.SOLVER.ID_EPSILON, bias=_biasON)  # new add by luo

    return classification


def check_loss_type_valid(loss_type_str):
    valid = True
    s = loss_type_str.split('_')
    for idx in range(len(s) - 1):
        try:
            curr_ = d_l[s[idx]]
            next_ = d_l[s[idx + 1]]
            if curr_ > next_:
                valid = False
        except KeyError:
            valid = False
    return valid


def extract_loss_names(loss_type_str):
    assert check_loss_type_valid(loss_type_str), "Wrong loss format"
    return loss_type_str.split('_')


# TODO: Make this into an torch module
def make_loss_with_center(cfg, num_classes):  # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34' or 'osnet' in cfg.MODEL.NAME:
        feat_dim = 512
    else:
        feat_dim = 2048

    loss_sequences = extract_loss_names(cfg.MODEL.METRIC_LOSS_TYPE)

    print("Criterions: ")
    print("-----------------")
    if 'center' in loss_sequences:
        center_criterion = center_loss(cfg, num_classes, feat_dim).cuda()
    if 'triplet' in loss_sequences:
        triplet = triplet_loss(cfg, num_classes, feat_dim)
    if 'CTL' in loss_sequences:
        ctl = CTL(cfg, num_classes, feat_dim)

    xent = id_loss(loss_sequences, cfg, num_classes, feat_dim).cuda()
    print(xent)
    print("numclasses:", num_classes)

    def loss_func(out_feat, global_feat, targets):
        loss_components = {}
        loss_center = cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(global_feat, targets)
        loss_classification = cfg.SOLVER.CLASSIFICATION_LOSS_WEIGHT * xent(out_feat, targets)
        loss_components['center'] = loss_center
        loss_components['classification'] = loss_classification
        loss_total = loss_center + loss_classification

        if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
            loss_triplet, dist_ap, dist_an = triplet(global_feat, targets)
            loss_components['triplet'] = loss_triplet
            loss_components['dist_ap'] = dist_ap.detach().mean()
            loss_components['dist_an'] = dist_an.detach().mean()
            loss_total += loss_triplet
        if 'CTL' in cfg.MODEL.METRIC_LOSS_TYPE:
            loss_ctl = ctl(global_feat, targets)
            loss_components['CTL'] = loss_ctl
            loss_total += loss_ctl
        # else:
        #     raise ValueError('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
        #                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

        return loss_total, loss_components

    return loss_func, center_criterion, xent
