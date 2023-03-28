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

    def __init__(self, in_features, s=30., m=0.35, num_classes=625, epsilon=0.1):
        super(AMSoftmaxLoss, self).__init__()
        self.in_features = in_features
        self.s = s
        self.m = m
        self.num_classes = num_classes
        self.CrossEntropy = CrossEntropyLabelSmooth(self.num_classes, epsilon=epsilon)
        # TODO: Remove this .cuda() after making the combined loss function into an module
        self.fc = nn.Linear(self.in_features, self.num_classes, bias=False).cuda()
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


class CurricularFace(nn.Module):
    """
    CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition
    Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, Feiyue Huang

    Code by the original author: HuangYG123@github.com
    """
    def __init__(self, in_features, out_features, m=0.5, s=64.):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = nn.Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    @staticmethod
    def l2_norm(input, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)

        return output

    def forward(self, embbedings, label):
        embbedings = self.l2_norm(embbedings, axis=1)
        kernel_norm = self.l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

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
        return output, origin_cos * self.s
