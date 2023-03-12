import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, (3, 7), 1, padding=1, dilation=1)
    
    def forward(self, x):
        y = self.conv(x)
        return y


def check_receptivefield(net, input_sz):
    b, c, w, h = input_sz
    x = torch.ones(*[b, c, w, h], requires_grad=True)
    y = net(x)
    grad = y[:, :, w//2, h//2].abs().sum().backward()
    # print(x.grad)
    idx = torch.nonzero(x.grad > 0)
    i1 = max(idx[:, -2])-min(idx[:, -2])
    i2 = max(idx[:, -1])-min(idx[:, -1])
    rf = (i1, i2)
    return rf


def conv2matmul(x, w):
    pass


if __name__ == "__main__":
    net = Net()
    input_sz = (1, 3, 32, 128)
    rf = check_receptivefield(net, input_sz)
    print(rf)