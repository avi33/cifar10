import torch

def fda(x, lambda_max=0.1):
    b, c, h, w = x.shape
    idx = torch.randperm(b)
    X = torch.fft.rfft2(x, dim=(-2, -1))    
    A = X.abs()
    A_perm = A[idx].clone()
    
    _, _, H, W = X.shape
    #extract low freq
    W *= 2
    lam = torch.rand(1, device=x.device) * lambda_max
    k = torch.floor(min(H, W)*lam * 0.5).int()        
    A[:, :, :k, :k] = A_perm[:, :, :k, :k].clone()
    A[:, :, H-k+1:H, :k] = A_perm[:, :, H-k+1:h, :k]
    X = A*torch.exp(1j*X.angle())
    x = torch.fft.ifft2(X, dim=(-2, -1), s=(h, w)).real
    return x

if __name__ == "__main__":
    pass