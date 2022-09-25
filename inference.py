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
    parser.add_argument("--f_res", default='outputs/t1', type=Path)    
    args = parser.parse_args()
    return args

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
    from modules import Net
    chkpnt = torch.load(args.f_res / 'chkpnt.pt', map_location=torch.device(device))
    net = Net()
    net.load_state_dict(chkpnt['model_dict'], strict=True)
    net.eval()    
    net.to(device)
    del chkpnt
    acc = 0
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_est = net(x)
        acc += accuracy(y_est, target=y, topk=(1,))[0]
    acc = acc/len(test_loader)
    print(acc)


def inference_cifar1class():    
    chosen_label = random.randint(0, 10)
    print(chosen_label)
    args = parse_args()
    f_res = args.f_res
    with open(f_res / "args.yml", "r") as f:
        args = yaml.load(f, yaml.Loader)
    args.f_res = f_res
    test_augs = T.Compose([T.ToTensor(), 
                        T.Normalize([0.5]*3, [0.5]*3)])
    test_set = Dataset(train=False, transform=test_augs, download=True, root='data')
    test_loader = DataLoader(test_set, 
                            batch_size=1,#args.batch_size, 
                            shuffle=False, 
                            drop_last=False, 
                            num_workers=4, 
                            pin_memory=True)
    from modules import Net, ImplicitLayer
    chkpnt = torch.load(args.f_res / 'chkpnt.pt', map_location=torch.device(device))
    net = Net()
    net.load_state_dict(chkpnt['model_dict'], strict=True)
    net.eval()
    net.to(device)
    del chkpnt
    acc = 0
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        y_est = net(x)
        y = chosen_label == y
        y_est = y_est.softmax(-1)[:, chosen_label] > 0.5
        acc += (1.*(y_est.data == y)).mean()
        # acc += accuracy(y_est, target=y, topk=(1,))[0]
    acc = acc/len(test_loader)*100
    print(acc)
    
    chkpnt = torch.load(args.f_res / 'chkpnt.pt', map_location=torch.device(device))
    net = ImplicitLayer(chkpnt['model_dict'], max_iter=10)
    net.eval()
    net.to(device)
    del chkpnt
    acc = 0
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        # y = y.to(device)
        y_est = net(x)
        y = chosen_label == y
        y_est = y_est.softmax(-1)[:, chosen_label] > 0.5
        acc += (1.*(y_est.data.cpu() == y)).mean()
        # acc += accuracy(y_est, target=y, topk=(1,))[0]
    acc = acc/len(test_loader)*100
    print(acc)

if __name__ == "__main__":
    inference_cifar()