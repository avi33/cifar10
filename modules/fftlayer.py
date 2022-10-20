from re import T
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from modules import AntiAliasDownsampleLayer, FastGlobalAvgPool2d

def create_net(args):
    net = Net()
    return net
    
class FFT(nn.Module):
    def __init__(self, c_in) -> None:
        super().__init__()
        self.fft = torch.fft.fft2
        self.ifft = torch.fft.ifft2
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2*c_in, c_in, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c_in),
            nn.LeakyReLU(0.2, True)
            )

    def forward(self, x):
        x = self.fft(x)
        x = torch.cat((x.real, x.imag), dim=1)
        x = self.block(x)
        x = self.ifft(x).real
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        model = []
        model += [FFT(3)]
        nf = 32        
        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=3, out_channels=nf, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True)
            ]                
        for k in range(4):
            nf_out = int(1.6*nf)
            model += [            
                nn.ReflectionPad2d(1),
                nn.Conv2d(nf, nf_out, 3, stride=1),
                nn.BatchNorm2d(nf_out),
                nn.LeakyReLU(0.2, True),
                AntiAliasDownsampleLayer(channels=nf_out),                
            ]
            nf = nf_out
        
        self.backbone = nn.Sequential(*model)             
        model = [
            FastGlobalAvgPool2d(flatten=True),
            nn.Linear(nf, 10)
        ]
        self.dense = nn.Sequential(*model)
        
    def forward(self, x):        
        x = self.backbone(x)
        y = self.dense(x)
        return y

if __name__ == "__main__":
    x = torch.randn(2, 3, 64, 64).requires_grad_(True).cuda()
    FF = Net().cuda()#FFT(3, 16)
    y = FF(x)
    print(y.shape, y.grad_fn)