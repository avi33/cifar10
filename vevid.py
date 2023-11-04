import numpy as np
import torch
import torch.nn as nn
import torchvision
from kornia.color import hsv_to_rgb, rgb_to_hsv
from torch.fft import fft2, fftshift, ifft2
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VEVID(nn.Module):
    def __init__(self, S, T, b, G, h=None, w=None):
        super().__init__()
        """initialize the VEVID GPU version class

        Args:            
            h (int, optional): height of the image to be processed. Defaults to None.
            w (int, optional): width of the image to be processed. Defaults to None.
        """
        self.h = h
        self.w = w        
        self.init_kernel(S=S, T=T)
        self.b = b
        self.G = G
        
    @staticmethod
    def cart2pol_torch(x, y):
        """convert cartesian coordiates to polar coordinates with PyTorch

        Args:
            x (torch.Tensor): cartesian coordinates in x direction
            y (torch.Tensor): cartesian coordinates in x direction

        Returns:
            tuple: polar coordinates theta and rho
        """
        theta = torch.atan2(y, x)
        rho = torch.hypot(x, y)
        return (theta, rho)
    
    @staticmethod
    def preprocess(img):
        return rgb_to_hsv(img)
        
    def init_kernel(self, S, T):
        """initialize the phase kernel of VEViD

        Args:
            S (float): phase strength
            T (float): variance of the spectral phase function
        """
        # create the frequency grid
        u = torch.linspace(-0.5, 0.5, self.h).float()
        v = torch.linspace(-0.5, 0.5, self.w).float()
        [U, V] = torch.meshgrid(u, v, indexing="ij")
        # construct the kernel
        [self.THETA, self.RHO] = self.cart2pol_torch(U, V)
        vevid_kernel = torch.exp(-self.RHO**2 / T)
        vevid_kernel = (vevid_kernel / torch.max(abs(vevid_kernel))) * S
        self.register_buffer('vevid_kernel', vevid_kernel)
        

    def apply_kernel(self, img, color=False, lite=False):
        """apply the phase kernel onto the image

        Args:
            b (float): regularization term
            G (float): phase activation gain
            color (bool, optional): whether to run color enhancement. Defaults to False.
            lite (bool, optional): whether to run VEViD lite. Defaults to False.
        """
        if color:
            channel_idx = 1
        else:
            channel_idx = 2
        vevid_input = img[channel_idx, :, :]
        if lite:
            vevid_phase = torch.atan2(-self.G * (vevid_input + b), vevid_input)
        else:
            vevid_input_f = fft2(vevid_input + self.b)
            img_vevid = ifft2(
                vevid_input_f * fftshift(torch.exp(-1j * self.vevid_kernel))
            )
            vevid_phase = torch.atan2(self.G * torch.imag(img_vevid), vevid_input)
        vevid_phase_norm = (vevid_phase - vevid_phase.min()) / (
            vevid_phase.max() - vevid_phase.min()
        )
        img[channel_idx, :, :] = vevid_phase_norm
        vevid_output = hsv_to_rgb(img)
        return vevid_output
    
    def forward(self, img, color=False, lite=False):        
        """run the full VEViD algorithm
        Args:
            img_file (str): path to the image
            S (float): phase strength
            T (float): variance of the spectral phase function
            b (float): regularization term
            G (float): phase activation gain
            color (bool, optional): whether to run color enhancement. Defaults to False.

        Returns:
            torch.Tensor: enhanced image
        """
        img = self.preprocess(img=img)        
        img_enh = self.apply_kernel(img, color, lite=lite)

        return img_enh
    

if __name__ == "__main__":
    img = Image.open(r"")
    w, h = img.size
    S = 0.2
    T = 0.001
    b = 0.16
    G = 1.4
    V = VEVID(S=S, T=T, w=w, h=h, b=b, G=G)
    V.to(device)
    img_t = torchvision.transforms.ToTensor()(img)
    img_t = img_t.expand(3, -1, -1)
    img_enh = V(img_t.to(device)).cpu()
    fig, ax = plt.subplots(2)
    ax[0].imshow(img)
    ax[1].imshow(img_enh.permute(1, 2, 0))
    plt.show()
    (torchvision.transforms.ToPILImage()(img_enh)).save("tmp.png")