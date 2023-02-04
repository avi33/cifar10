from copy import deepcopy
import os
from statistics import mode
import torch
from torchvision.datasets import CIFAR10 as Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import argparse
from pathlib import Path
from logger import accuracy
import yaml
import random
from helper_funcs import save_torchscript_model, load_torchscript_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_res", default='outputs/quant', type=Path)
    args = parser.parse_args()
    return args

def inference_cifar():
    global device
    args = parse_args() 
    f_res = args.f_res
    with open(f_res / "args.yml", "r") as f:
        args = yaml.load(f, yaml.Loader)
    args.f_res = f_res
    test_augs = T.Compose([T.ToTensor(), 
                           T.Normalize([0.5]*3, [0.5]*3)])
    test_set = Dataset(train=False, transform=test_augs, download=False, root='data')
    test_loader = DataLoader(test_set, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            drop_last=False, 
                            num_workers=4, 
                            pin_memory=True)
    
    from modules.models import Net    
    net = Net(emb_dim=128, n_classes=args.n_classes, nf=16, tf_type=args.tf_type, factors=[2, 2, 2], inp_sz=(32, 32))
    # from modules.fftlayer import Net
    # net = Net(nf=16)
    net.eval()
    net.to(device)
    chkpnt = torch.load(args.f_res / 'chkpnt.pt', map_location=torch.device(device))
    net.load_state_dict(chkpnt['model_dict'], strict=True)    
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