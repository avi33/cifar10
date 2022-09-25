import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.padding import ReflectionPad1d
from torch.nn.utils import weight_norm
import torch.nn.init as init

def create_net(args):
    net = Net()
    return net

class Sine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        x = (x-x_min) / (x_max-x_min+1e-4)
        x = torch.sin(1e-4+x*np.pi/2-1e-4)
        return x

class FastGlobalAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1)

class AntiAliasDownsampleLayer(nn.Module):
    def __init__(self, remove_aa_jit: bool = False, filt_size: int = 3, stride: int = 2,
                 channels: int = 0):
        super(AntiAliasDownsampleLayer, self).__init__()
        if not remove_aa_jit:
            self.op = DownsampleJIT(filt_size, stride, channels)
        else:
            self.op = Downsample(filt_size, stride, channels)

    def forward(self, x):
        return self.op(x)


@torch.jit.script
class DownsampleJIT(object):
    def __init__(self, filt_size: int = 3, stride: int = 2, channels: int = 0):
        self.stride = stride
        self.filt_size = filt_size
        self.channels = channels

        assert self.filt_size == 3
        assert stride == 2
        a = torch.tensor([1., 2., 1.])

        filt = (a[:, None] * a[None, :]).clone().detach()
        filt = filt / torch.sum(filt)
        self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1)).cuda().half()

    def __call__(self, input: torch.Tensor):
        if input.dtype != self.filt.dtype:
            self.filt = self.filt.float() 
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=2, padding=0, groups=input.shape[1])

class Downsample(nn.Module):
    def __init__(self, filt_size=3, stride=2, channels=None):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        assert self.filt_size == 3
        a = torch.tensor([1., 2., 1.])

        filt = (a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)

        # self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])


class ResBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(            
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(dim, dim, kernel_size=3, dilation=dilation, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, dim, kernel_size=1),            
        )
        self.shortcut = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        nf = 32        
        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=3, out_channels=nf, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=0, bias=False),
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
    b = 2
    x = torch.randn(b, 3, 64, 64).cuda()
    net = Net().cuda()
    y = net(x)
    print(y.shape)
    #torch.save(net.state_dict(), "ff.pt")