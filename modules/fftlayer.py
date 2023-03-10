from re import T
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from modules.anti_aliasing_downsample import Down
from modules.average_pooling import FastGlobalAvgPool
from modules.res_block import ResBlock

def create_net(args):
    net = Net()
    return net
    
class FFTConv(nn.Module):
    def __init__(self, c_in) -> None:
        super().__init__()
        self.fft = torch.fft.rfft2
        self.ifft = torch.fft.irfft2
        self.f_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2*c_in, c_in, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c_in),
            nn.LeakyReLU(0.2, True)
            )
        self.t_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c_in, c_in, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c_in),
            nn.LeakyReLU(0.2, True)
            )
        self.post = nn.Conv2d(2*c_in, c_in, 1, 1)

    def forward(self, x):
        f = self.fft(x)
        f = torch.cat((f.real, f.imag), dim=1)
        f = self.f_block(f)
        f = self.ifft(f).real
        t = self.t_block(x)
        x = self.post(torch.cat((t, f), dim=1))
        return x

class Net(nn.Module):
    def __init__(self, nf=16):
        super().__init__()
        model = []
        model += [FFTConv(3)]
        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=3, out_channels=nf, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True)
            ]                
        for k in range(3):            
            model += [Down(nf=nf, kernel_size=3, stride=2)]
            nf *= 2            
            # model += [ResBlock(dim=nf, dilation=1),
            #         #   ResBlock(dim=nf, dilation=3),
            #         ]
            model += [FFTConv(nf)]
        
        self.backbone = nn.Sequential(*model)             
        model = [
            FastGlobalAvgPool(flatten=True),
            nn.Linear(nf, 10)
        ]
        self.dense = nn.Sequential(*model)
        
    def forward(self, x):        
        x = self.backbone(x)
        y = self.dense(x)
        return y

if __name__ == "__main__":
    from helper_funcs import count_parameters, measure_inference_time
    x = torch.randn(2, 3, 32, 32).requires_grad_(True).cuda()
    net = Net(nf=16).cuda()#FFT(3, 16)
    y = net(x)
    print(y.shape, y.grad_fn)
    print(count_parameters(net)/1e6)
    t = measure_inference_time(net, x)        
    print("inference time :{}+-{}".format(t[0], t[1]))