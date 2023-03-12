import torch
import torch.nn as nn


class SpaceToDepth1d(nn.Module):
    def __init__(self, block_size=2):
        super().__init__()        
        self.bs = block_size        
    
    def forward(self, x):
        b, c, n = x.size()
        unfolded_x = torch.nn.functional.unfold(x, self.bs, stride=self.bs)
        return unfolded_x.view(n, c * self.bs * 2, n // self.bs)
    

@torch.jit.script
class SpaceToDepth1dJit(object):
    def __init__(self, block_size: int) -> None:
        self.bs = block_size

    def __call__(self, x: torch.Tensor):
        b, c, n = x.size()
        unfolded_x = torch.nn.functional.unfold(x, self.bs, stride=self.bs)
        return unfolded_x.view(n, c * self.bs * 2, n // self.bs)


class SpaceToDepth1dModule(nn.Module):
    def __init__(self, bs=2, remove_model_jit=False):
        super().__init__()
        if not remove_model_jit:
            self.op = SpaceToDepth1dJit(bs)
        else:
            self.op = SpaceToDepth1d(bs)

    def forward(self, x):
        return self.op(x)