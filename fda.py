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
    print(lam, idx)
    k = torch.floor(min(H, W)*lam * 0.5).int()        
    A[:, :, :k, :k] = A_perm[:, :, :k, :k].clone()
    # A[:, :, H-k+1:H, :k] = A_perm[:, :, H-k+1:h, :k].clone()
    X = A*torch.exp(1j*X.angle())
    x = torch.fft.irfft2(X, dim=(-2, -1), s=[H, W])
    return x

if __name__ == "__main__":
    from PIL import Image
    im1 = Image.open(r"/media/avi/8E56B6E056B6C86B/datasets/accutures/benshemen29_12_22/recording_1/benshemen291222_r_1_f_1_rgb.png")
    im2 = Image.open(r"/media/avi/8E56B6E056B6C86B/datasets/accutures/benshemen29_12_22/recording_13/benshemen291222_r_13_f_1_rgb.png")
    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    i1 = T.ToTensor()(im1)
    i2 = T.ToTensor()(im2)
    x = torch.cat((i1.unsqueeze(0), i2.unsqueeze(0)), dim=0)
    y = fda(x, 1)
    fig, ax = plt.subplots(2)
    ax[0].imshow(im1)
    ax[1].imshow(y[0].permute(1, 2, 0))
    plt.show()
