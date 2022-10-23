import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
class DownsampleJIT(object):
    def __init__(self, filt_size: int = 3, stride: int = 2, channels: int = 0, device: str = "cuda"):
        self.stride = stride
        self.filt_size = filt_size
        self.channels = channels

        assert self.filt_size == 3
        assert stride == 2
        
        device = torch.device(device)

        a = torch.tensor([1., 2., 1.])

        filt = (a[:, None] * a[None, :]).clone().detach()
        filt = filt / torch.sum(filt)
        self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1)).to(device).half()

    def __call__(self, input: torch.Tensor):
        if input.dtype != self.filt.dtype:
            self.filt = self.filt.float() 
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=2, padding=0, groups=input.shape[1])

class Downsample(nn.Module):
    def __init__(self, filt_size=3, stride=2, channels=None, device="cuda"):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        # assert self.filt_size == 3
        device = torch.device(device)
        ha = torch.arange(1, filt_size//2+1+1, 1)
        a = torch.cat((ha, ha.flip(dims=[-1,])[1:])).float()
        a = a / a.sum()


        a = torch.tensor([1., 2., 1.])

        filt = (a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1)).to(device).half()
        self.register_buffer('filt', filt)

    def forward(self, input):
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])

class AntiAliasDownsampleLayer(nn.Module):
    def __init__(self, remove_aa_jit: bool = False, filt_size: int = 3, stride: int = 2,
                 channels: int = 0, device: str = "cuda"):
        super(AntiAliasDownsampleLayer, self).__init__()
        if not remove_aa_jit:
            self.op = DownsampleJIT(filt_size, stride, channels, device=device)
        else:
            self.op = Downsample(filt_size, stride, channels, device=device)

    def forward(self, x):
        return self.op(x)