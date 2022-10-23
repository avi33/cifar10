import copy
import torch

def simple_quant(net, n_bits=None):        
    netq = copy.deepcopy(net)
    for name, p in netq.named_parameters():
        p.data = torch.round(p.data * 2**n_bits) / 2**n_bits
    return netq