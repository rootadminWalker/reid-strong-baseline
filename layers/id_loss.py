import torch
from torch import nn


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class AMSoftmaxLoss(nn.Module):
    """
    Original code by ppriyank@github.com
    """
    def __init__(self, s=30, m=0.35, num_classes=625, use_gpu=True, epsilon=0.1):
        super(AMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.CrossEntropy = CrossEntropyLabelSmooth(self.num_classes, epsilon=epsilon, use_gpu=use_gpu)

    def forward(self, features, labels):
        '''
        x : feature vector : (b x  d) b= batch size d = dimension
        labels : (b,)
        classifier : Fully Connected weights of classification layer (dxC), C is the number of classes: represents the vectors for class
        '''
        # x = torch.rand(32,2048)
        # label = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,])
        # features = nn.functional.normalize(features, p=2, dim=1)  # normalize the features
        # with torch.no_grad():
        #     classifier.weight.div_(torch.norm(classifier.weight, dim=1, keepdim=True))
        #
        # cos_angle = classifier(features)
        cos_angle = features
        cos_angle = torch.clamp(cos_angle, min=-1, max=1)
        b = features.size(0)
        for i in range(b):
            cos_angle[i][labels[i]] = cos_angle[i][labels[i]] - self.m
        weighted_cos_angle = self.s * cos_angle
        log_probs = self.CrossEntropy(weighted_cos_angle, labels)
        return log_probs
