import torch
import torch.nn as nn


class ConvFFT(nn.Module):
    def __init__(self, nf, factors=[2, 2, 2]) -> None:
        from modules.components.fftlayer import FFTConv
        super().__init__()        
        block = [
            nn.Conv2d(3, nf, 3, 1, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True)            
        ]
        nf = 16
        for _, f in enumerate(factors):
            block += [FFTConv(c_in=nf), Down(nf, kernel_size=f+1, stride=f)]
            nf *= 2        
        self.block = nn.Sequential(*block)
            
    def forward(self, x):
        x = self.block(x)
        return x