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
    
    from modules.modules import Net, NetQ    
    net = Net()    
    if args.quant:
        device = torch.device("cpu")
        # net = NetQ(model_flp=net).to(device)                
        # net.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        # net = torch.quantization.convert(net, inplace=True)
        net = load_torchscript_model(args.f_res / 'net_q_jit.pt', device=device)
        net.eval()
        # net = chkpnt['model_dict_quant']
        # net.to(device)        
        # net.quant.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        # torch.quantization.prepare_qat(net, inplace=True)                
        # chkpnt = torch.load(args.f_res / 'chkpnt.pt', map_location=torch.device(device))        
        # net = torch.quantization.convert(net, inplace=True)
        # net.load_state_dict(chkpnt['model_dict'], strict=True)
        # net.eval()
        # save_torchscript_model(model=net, model_dir=args.save_path, model_filename="model_quant_jit.pt")
        # net = load_torchscript_model(model_filepath=os.path.join(args.save_path, "model_quant_jit.pt"), device=device)
    else:
        net.eval()
        net.to(device)
        chkpnt = torch.load(args.f_res / 'chkpnt.pt', map_location=torch.device(device))
        net.load_state_dict(chkpnt['model_dict'], strict=True)
        del chkpnt
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