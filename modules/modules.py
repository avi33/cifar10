from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from modules.space_to_depth import SpaceToDepthModule
from modules.average_pooling import FastGlobalAvgPool2d
from modules.anti_aliasing_downsample import AntiAliasDownsampleLayer
from modules.res_block import ResBlock

def create_net(args):
    net = Net(args)    
    return net

class Net(nn.Module):
    def __init__(self):
        super().__init__()        
        nf = 32        
        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=3, out_channels=nf, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True),
            # ResBlock(nf),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True),
            # ResBlock(nf),

            ]
        fac = 2
        for k in range(4):
            nf_out = int(fac*nf)
            model += [            
                nn.ReflectionPad2d(1),
                nn.Conv2d(nf, nf_out, 3, stride=2, bias=False),
                nn.BatchNorm2d(nf_out),
                nn.LeakyReLU(0.2, True),
                # nn.Conv2d(nf_out, nf_out, 3, stride=2, padding=1),
                # AntiAliasDownsampleLayer(channels=nf_out),
                # ResBlock(nf_out),
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

class NetQ(nn.Module):
    def __init__(self, model_flp) -> None:
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model_flp = model_flp
        
    def forward(self, x):
        x = self.quant(x)
        x = self.model_flp(x)        
        y = self.dequant(x)
        return y

if __name__ == "__main__":
    b = 2
    x = torch.randn(b, 3, 64, 64).cuda()
    net = Net().cuda()
    y = net(x)
    print(y.shape)