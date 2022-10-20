from copy import deepcopy
import torch
from torchvision.datasets import CIFAR10 as Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import argparse
from pathlib import Path
from logger import accuracy
import yaml
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_res", default='outputs/t2-label_smooth', type=Path)    
    args = parser.parse_args()
    return args

def simple_quant(net, n_bits=None):        
    netq = deepcopy(net)
    for name, p in netq.named_parameters():
        p.data = torch.round(p.data * 2**n_bits) / 2**n_bits
    return netq
        
def inference_cifar():
    args = parse_args()
    f_res = args.f_res
    with open(f_res / "args.yml", "r") as f:
        args = yaml.load(f, yaml.Loader)
    args.f_res = f_res
    test_augs = T.Compose([T.ToTensor(), 
                           T.Normalize([0.5]*3, [0.5]*3)])
    test_set = Dataset(train=False, transform=test_augs, download=True, root='data')
    test_loader = DataLoader(test_set, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            drop_last=False, 
                            num_workers=4, 
                            pin_memory=True)
    from modules.modules import Net
    chkpnt = torch.load(args.f_res / 'chkpnt.pt', map_location=torch.device(device))
    net = Net()
    net.load_state_dict(chkpnt['model_dict'], strict=True)
    net.eval()    
    net.to(device)
    del chkpnt
    net = simple_quant(net, 16)
    acc = 0
    for i, (x, y) in enumerate(test_loader):
        if i % 10 == 0:
            print("{}/{}".format(i, len(test_loader)))
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_est = net(x)
        acc += accuracy(y_est, target=y, topk=(1,))[0]
    acc = acc/len(test_loader)
    print(acc)


if __name__ == "__main__":
    inference_cifar()