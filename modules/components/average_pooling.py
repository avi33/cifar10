import torch
import torch.nn as nn

class FastGlobalAvgPool(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1)
        
if __name__ == "__main__":
    # Example usage
    x = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    pool = FastGlobalAvgPool(flatten=True)
    output = pool(x)
    print(output.shape)  # Should be [1, 3] if flatten is True