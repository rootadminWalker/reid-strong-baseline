import math

import torch
import torch.nn.functional as F
from torch import nn

from utils import weights_init_classifier


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = F.one_hot(targets, num_classes=self.num_classes).float()
        # targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class CrossEntropyHead(nn.Module):
    def __init__(self, in_features, num_classes, epsilon=0., bias=False):
        super(CrossEntropyHead, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.CrossEntropy = CrossEntropyLabelSmooth(self.num_classes, epsilon)
        self.fc = nn.Linear(self.in_features, self.num_classes, bias=bias)
        self.fc.apply(weights_init_classifier)

    def forward(self, x, labels):
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.num_classes

        features = self.fc(x)
        return self.CrossEntropy(features, labels)


class AMSoftmaxLoss(nn.Module):
    """
    Original code by ppriyank@github.com
    """

    def __init__(self, in_features, num_classes=625, s=30., m=0.35, epsilon=0.0):
        super(AMSoftmaxLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.CrossEntropy = CrossEntropyLabelSmooth(self.num_classes, epsilon=epsilon)
        # TODO: Remove this .cuda() after making the combined loss function into an module
        self.fc = nn.Linear(self.in_features, self.num_classes, bias=False)
        self.fc.apply(weights_init_classifier)

    def forward(self, x, labels):
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.num_classes

        x = F.normalize(x, p=2, dim=1)

        with torch.no_grad():
            self.fc.weight.div_(torch.norm(self.fc.weight, dim=1, keepdim=True))

        b = x.size(0)
        features = self.fc(x)
        features = torch.clamp(features, min=-1, max=1)
        for i in range(b):
            features[i][labels[i]] = features[i][labels[i]] - self.m
        s_features = self.s * features
        log_probs = self.CrossEntropy(s_features, labels)
        return log_probs


class ArcFace(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)

        Original code by ronghuaiyang@github.com in arcface-pytorch
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, epsilon=0.0, easy_margin=False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.cross_entropy = CrossEntropyLabelSmooth(num_classes=self.out_features, epsilon=self.epsilon)

    def get_cosine(self, x):
        return F.linear(F.normalize(x), F.normalize(self.weight))

    def forward(self, x, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = self.get_cosine(x)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        loss = self.cross_entropy(output, label)
        return loss


class CurricularFace(nn.Module):
    """
    CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition
    Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, Feiyue Huang

    Original code by HuangYG123@github.com
    """

    def __init__(self, in_features, out_features, s=64., m=0.5, epsilon=0.0):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.epsilon = epsilon
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = nn.Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

        self.cross_entropy = CrossEntropyLabelSmooth(num_classes=self.out_features, epsilon=self.epsilon)

    @staticmethod
    def l2_norm(input, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)

        return output

    def get_cosine(self, x):
        x = self.l2_norm(x, axis=1)
        kernel_norm = self.l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(x, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        return cos_theta

    def forward(self, x, label):
        cos_theta = self.get_cosine(x)
        target_logit = cos_theta[torch.arange(0, x.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s

        loss = self.cross_entropy(output, label)
        return loss


class SubCenterArcFace(ArcFace):
    def __init__(self, in_features, out_features, s=64., m=0.5, epsilon=0.0, easy_margin=False, K=3):
        super(SubCenterArcFace, self).__init__(
            in_features=in_features,
            out_features=out_features * K,
            s=s,
            m=m,
            epsilon=epsilon,
            easy_margin=easy_margin
        )
        self.out_features = out_features
        self.K = K
        assert self.K > 1, "You should have more than 1 sub-centers in Sub-center ArcFace." \
                           "If you don't want any sub-centers, use ArcFace instead."

        self.cross_entropy = CrossEntropyLabelSmooth(num_classes=self.out_features, epsilon=self.epsilon)

    def get_cosine(self, x):
        cosine = super(SubCenterArcFace, self).get_cosine(x)
        cosine = cosine.view(-1, self.out_features, self.K)
        cosine, _ = cosine.max(axis=2)
        return cosine


class SubCenterCurricularFace(CurricularFace):
    def __init__(self, in_features, out_features, s=64., m=0.5, epsilon=0.0, K=3):
        super(SubCenterCurricularFace, self).__init__(
            in_features=in_features,
            out_features=out_features * K,
            s=s,
            m=m,
            epsilon=epsilon,
        )
        self.out_features = out_features
        self.K = K
        assert self.K > 1, "You should have more than 1 sub-centers in Sub-center ArcFace." \
                           "If you don't want any sub-centers, use ArcFace instead."

        self.cross_entropy = CrossEntropyLabelSmooth(num_classes=self.out_features, epsilon=self.epsilon)

    def get_cosine(self, x):
        cos_theta = super(SubCenterCurricularFace, self).get_cosine(x)
        cos_theta = cos_theta.view(-1, self.out_features, self.K)
        cos_theta, _ = cos_theta.max(axis=2)
        return cos_theta
