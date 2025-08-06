import torch
import torch.nn as nn
import torch.nn.functional as F

    
class FFTConv2d(nn.Module):
    def __init__(self, c_in, c_out):
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
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.2, True)
            )
        self.post = nn.Conv2d(2*c_in, c_out, 1, 1)

    def forward(self, x):
        f = self.fft(x)
        f = torch.cat((f.real, f.imag), dim=1)
        f = self.f_block(f)
        f = self.ifft(f).real
        t = self.t_block(x)
        x = self.post(torch.cat((t, f), dim=1))
        return x


if __name__ == "__main__":
    x = torch.randn(1, 3, 32, 32)
    conv_fft = FFTConv2d(3)
    out = conv_fft(x)
    print(out.shape)  # Should be [1, 3, 32, 32]