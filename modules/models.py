import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.components.anti_aliasing_downsample import Down
from modules.components.res_block import ResBlock
from modules.components.average_pooling import FastGlobalAvgPool
from modules.components.space_to_depth_2d import SpaceToDepth

class CNN(nn.Module):
    def __init__(self, nf, factors=[2, 2, 2]) -> None:
        super().__init__()
        block = [
            SpaceToDepth(),
            nn.Conv2d(3 * 16, nf, 3, 1, 1, bias=False),                        
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
    def __init__(self, emb_dim, ff_dim, n_heads, n_layers, p, tf_type) -> None:
        super().__init__()
        self.emb_dim = emb_dim                        
        from modules.components.transformer_encoder_bn import TFEncoder
        from modules.components.fftlayer import FFTConv2d
        self.tf = TFEncoder(num_layers=n_layers, 
                            num_heads=n_heads, 
                            d_model=emb_dim, 
                            ff_hidden_dim=ff_dim, 
                            p=p, norm=nn.LayerNorm(emb_dim),
                            use_inner_pos_embedding=True)
        self.pos_emb = FFTConv2d(emb_dim, emb_dim)
        
        self.avg_pool = FastGlobalAvgPool(flatten=True)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):                
        x = self.pos_emb(x)
        x = x.view(x.shape[0], self.emb_dim, -1)
        x = self.tf(x)
        out = self.avg_pool(x)
        return out

class Net(nn.Module):
    def __init__(self, emb_dim, n_classes, nf, factors, tf_type, inp_sz) -> None:
        super().__init__()
        self.nf = nf
        self.cnn = CNN(nf=nf, factors=factors)
        self.tf = TFAggregation(emb_dim=emb_dim, ff_dim=emb_dim*4, n_heads=2, n_layers=4, p=0.1, tf_type=tf_type)                        
        self.project = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.tf(x)
        y = self.project(x)
        return y

if __name__ == "__main__":    
    pass