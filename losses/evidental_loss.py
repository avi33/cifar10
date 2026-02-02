import torch
from torch.nn import functional as F

class EvidentialLoss(torch.nn.Module):
    """
    Implements the evidential loss for classification tasks.
    Reference: "Evidential Deep Learning to Quantify Classification Uncertainty"
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
    

    def forward(self, outputs, labels):
        y_onehot = F.one_hot(labels, num_classes=outputs.shape[1])
        S = outputs.sum(dim=1, keepdim=True)
        p = outputs / S

        err = (y_onehot - p) ** 2
        var = p * (1 - p) / (S + 1)

        loss = (err + var).sum(dim=1)
        if self.reduction == "mean":        
            return loss.mean()
        elif self.reduction == "sum":        
            return loss.sum()
        else:
            return loss
        
def dirichlet_kl(alpha):
    K = alpha.size(1)
    prior = torch.ones_like(alpha)

    S = alpha.sum(dim=1, keepdim=True)
    S0 = prior.sum(dim=1, keepdim=True)

    lnB = torch.lgamma(S) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    lnB0 = torch.lgamma(S0) - torch.lgamma(prior).sum(dim=1, keepdim=True)

    digamma = torch.digamma(alpha)
    digamma_sum = torch.digamma(S)

    kl = ((alpha - prior)*(digamma - digamma_sum)).sum(dim=1, keepdim=True)
    kl = kl + lnB + lnB0

    return kl.mean()

if __name__ == "__main__":

    y = torch.tensor([0, 1, 2, 1])
    y_onehot = F.one_hot(y, num_classes=3).float()
    alpha = torch.tensor([[2.0, 1.0, 1.0],
                          [1.0, 3.0, 1.0],
                          [1.0, 1.0, 4.0],
                          [1.0, 2.0, 2.0]])
    E = EvidentialLoss(reduction="mean")
    loss = E(alpha, y_onehot)   
    print(f"loss={loss.item()}")
