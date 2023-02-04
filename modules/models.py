from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from modules.anti_aliasing_downsample import Down
from modules.res_block import ResBlock

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
    def __init__(self, seq_len, emb_dim, ff_dim, n_heads, n_layers, p, tf_type) -> None:
        super().__init__()
        self.emb_dim = emb_dim        
        if tf_type == "torch":
            tf_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=p, activation=F.relu, layer_norm_eps=1e-5, batch_first=True)
            self.tf = nn.TransformerEncoder(tf_layer, num_layers=n_layers, norm=nn.LayerNorm(emb_dim),)
        elif tf_type == "my":
            from modules.transformer_encoder_my import TFEncoder
            self.tf = TFEncoder(num_layers=n_layers, num_heads=n_heads, d_model=emb_dim, ff_hidden_dim=ff_dim, p=p, norm=nn.LayerNorm(emb_dim))
        self.cls_token = self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_emb = nn.Conv1d(seq_len+1, seq_len+1, 1, 1, groups=seq_len+1)
        self._reset_parameters()
        
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = x.view(x.shape[0], self.emb_dim, -1).transpose(2, 1).contiguous()
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_emb(x)
        x = self.tf(x)
        out = x[:, 0, :]
        return out

class Net(nn.Module):
    def __init__(self, emb_dim, n_classes, nf, factors, tf_type, inp_sz) -> None:
        super().__init__()
        self.nf = nf
        self.cnn = CNN(nf=16, factors=factors)
        seq_len = inp_sz[0]*inp_sz[1] // (np.prod(factors)**2)
        # self.proj_emb = nn.Conv2d(nf*2**len(factors), emb_dim, 1, 1, groups=emb_dim)
        self.tf = TFAggregation(seq_len=seq_len, emb_dim=emb_dim, ff_dim=emb_dim*4, n_heads=2, n_layers=4, p=0.1, tf_type=tf_type)                        
        self.project = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.tf(x)
        x = self.project(x)
        return x

if __name__ == "__main__":    
    from helper_funcs import count_parameters, measure_inference_time, check_receptivefield
    b = 1
    # x = torch.randn(b, 3, 64, 64).cuda()
    # net = CNN(nf=16).cuda()
    # y = net(x)
    # print(y.shape)

    # x = torch.randn(b, 128, 8, 8).cuda()
    # net = TFAggregation(64, 128, 128*4, 2, 4, 0.1, "torch").cuda()
    # y = net(x)
    # print(y.shape)

    x = torch.randn(b, 3, 32, 128).cuda()
    net = Net(nf=16, emb_dim=128, n_classes=10, tf_type="my", factors=[2, 2, 2], inp_sz=(32, 128)).cuda()
    y = net(x)
    print(y.shape)    
    print(count_parameters(net)/1e6)    
    t = measure_inference_time(net, x)        
    print("inference time :{}+-{}".format(t[0], t[1]))
    print(check_receptivefield(net, x))
    if False:
        #CIFAR
        x = torch.randn(b, 3, 32, 32).cuda()
        net = Net(nf=16, emb_dim=128, factors=[2, 2, 2], n_classes=10, tf_type="my", inp_sz=(32, 32)).cuda()
        y = net(x)
        print(count_parameters(net)/1e6)
        print(y.shape)
        t = measure_inference_time(net, x)
        print("inference time :{}+-{}".format(t[0], t[1]))
    
    if False:
        #COCO
        x = torch.randn(b, 3, 640, 480).cuda()
        net = Net(nf=16, emb_dim=128, factors=[2, 2, 4], n_classes=80, tf_type="my", inp_sz=(640, 480)).cuda()
        net.eval()
        y = net(x)
        print(y.shape)    
        print(count_parameters(net)/1e6)    
        t = measure_inference_time(net, x)        
        print("inference time :{}+-{}".format(t[0], t[1]))

        from RepVGG.repvggplus import create_RepVGGplus_by_name
        net = create_RepVGGplus_by_name("RepVGG-A1", deploy=True, use_checkpoint=False)    
        net.Linear = nn.Linear(1280, 80)
        net.eval()
        net.to("cuda")
        y = net(x)
        print(y.shape)