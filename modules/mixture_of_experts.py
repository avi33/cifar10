import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from modules.anti_aliasing_downsample import Down
from modules.res_block import ResBlock
from modules.average_pooling import FastGlobalAvgPool

class CNN(nn.Module):
    def __init__(self, nf, factors=[2, 2, 2]) -> None:
        super().__init__()
        block = [
            nn.Conv2d(3, nf, 5, 1, padding=2, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True)            
        ]
        for _, f in enumerate(factors):
            block += [Down(nf, kernel_size=f+1, stride=f)]
            nf *= 2
            block += [ResBlock(dim=nf, dilation=1)]
            block += [ResBlock(dim=nf, dilation=3)]
        self.block = nn.Sequential(*block)        
    
    def forward(self, x):
        x = self.block(x)        
        return x
    
class TFAggregation(nn.Module):
    def __init__(self, emb_dim, ff_dim, n_heads, n_layers, p, n_experts) -> None:
        super().__init__()
        self.emb_dim = emb_dim                
        # from modules.transformer_encoder_my import TFEncoder
        from modules.transformer_encoder_bn import TFEncoder
        self.tf = TFEncoder(num_layers=n_layers, 
                            num_heads=n_heads, 
                            d_model=emb_dim, 
                            ff_hidden_dim=ff_dim, 
                            p=p, norm=nn.LayerNorm(emb_dim),
                            use_inner_pos_embedding=True)
        self.pos_emb = nn.Conv2d(emb_dim, emb_dim, kernel_size=7, stride=1, padding=3, padding_mode='zeros', groups=emb_dim, bias=True)        
        self.cls_tocken_moe = nn.Parameter(torch.zeros(1, emb_dim, n_experts))
        self.n_experts = n_experts
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):                
        x = self.pos_emb(x)
        x = x.view(x.shape[0], self.emb_dim, -1)
        cls_tokens = self.cls_tocken_moe.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=2)
        x = self.tf(x)
        out=x[..., :self.n_experts]
        return out

class Net(nn.Module):
    def __init__(self, emb_dim, n_classes, nf, factors, tf_type, inp_sz) -> None:
        super().__init__()
        self.nf = nf
        self.cnn = CNN(nf=16, factors=factors)
        self.tf = TFAggregation(emb_dim=emb_dim, ff_dim=emb_dim*4, n_heads=2, n_layers=4, p=0.1, tf_type=tf_type)                        
        self.project = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.tf(x)
        y = self.project(x)
        return y
    
class MixtureOfExperts(nn.Module):
    def __init__(self, emb_dim, n_classes, nf, factors):
        super().__init__()
        self.cnn = CNN(nf=nf, factors=factors)
        self.tf = TFAggregation(emb_dim=emb_dim, ff_dim=emb_dim*4, n_heads=2, n_layers=4, p=0.1, n_experts=n_classes)        
        self.project = nn.Conv1d(emb_dim, 1, 1, 1)

        self.gate = nn.Conv1d(emb_dim, 1, 1, 1)

    def forward(self, x):
        x = self.cnn(x)        
        x = self.tf(x)
        gate = self.gate(x).squeeze(1)
        y = self.project(x)
        return y, gate
                
    
if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    x = x.to("cuda")
    M = MixtureOfExperts(emb_dim=128, n_classes=10, nf=16, factors=[2, 2, 2])
    M = M.to("cuda")
    y = M(x)
    print(y.shape)