import torch
import torch.nn as nn
import numpy as np

class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()
        
    def _norm_minmax(self, x):
        x_min = x.min(-1, keeddims=True)
        x_max = x.min(-1, keeddims=True)
        x.add_(-x_min).mul_(2*np.pi/(x_max-x_min))
        return

    def forward(self, x):
        self._norm_minmax(x)
        x = x + torch.sin(x)
        return x