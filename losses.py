import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
import numpy as np

class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                              self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

class LabelSmoothDirichletCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, alpha=None):
        with torch.no_grad():            
            if alpha is not None:
                targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(0.1 / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - 0.1)
            else:
                targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(0.1 / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - 0.1)
                
        return targets

    def forward(self, inputs, targets, alpha):        
        targets = LabelSmoothDirichletCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                              alpha)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

class ZLPR(nn.Module):
    def __init__(self, n_classes, is_multilabel=False, reduction="mean") -> None:
        super().__init__()
        self.reduction = reduction
        self.n_classes = n_classes
        self.is_multilabel = is_multilabel

    def forward(self, pred, target):            
        if self.is_multilabel:
            y = [F.one_hot(torch.Tensor(y_i).long(), self.n_classes).sum(dim=0).float() for y_i in y]
            y = torch.stack(y, dim=0).contiguous()
        else:            
            pred = torch.clamp(pred, min=-30, max=30)
            pred_trg = torch.gather(pred, 1, target.view(-1, 1))
            loss = torch.log(1+torch.exp(-pred_trg)) + torch.log(1+torch.exp(pred).sum(-1, keepdims=True)-torch.exp(pred_trg))
            if self.reduction == "sum":
                loss = loss.sum()
            elif self.reduction == "mean":
                loss = loss.mean()
            elif self.reduction == "none":
                pass
            else:
                raise ValueError("wrong reduction for loss {}".format(self.reduction))
        return loss

if __name__ == '__main__':
    L = ZLPR(10)
    pred = torch.randn(1, 10)
    target = torch.Tensor([2]).long()
    for k in range(10):
        pred_ = pred.clone()
        pred_[:, k] = 10        
        loss = L(pred_, target)
        print(loss)