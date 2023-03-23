import torch
from torch import nn
import torch.nn.functional as F
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
    def __init__(self, in_features, s=30, m=0.35, num_classes=625, epsilon=0.1):
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
