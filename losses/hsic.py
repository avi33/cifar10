import torch
import torch.nn as nn

class HSIC(nn.Module):
    def __init__(self, reduction="mean") -> None:
        super().__init__()
        self.reduction = reduction
    
    def forward(self, x, y, s_x=1, s_y=1):
        m,_ = x.shape #batch size
        K = HSIC.GaussianKernelMatrix(x,s_x)
        L = HSIC.GaussianKernelMatrix(y,s_y)
        H = torch.eye(m, device=x.device) - 1.0/m * torch.ones((m,m), device=x.device)
        #H = H.double().cuda()
        hsic = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
        if self.reduction == 'sum':
            hsic *= m
        return hsic
            

    @staticmethod
    def GaussianKernelMatrix(x, sigma=1):
        pairwise_distances_ = HSIC.pairwise_distances(x)
        return torch.exp(-pairwise_distances_ /sigma)

    @staticmethod
    def pairwise_distances(x):
        #x should be two dimensional
        instances_norm = torch.sum(x**2,-1).reshape((-1,1))
        return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()