import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import ToPILImage
import yaml
from torchvision.datasets import CIFAR10 as Dataset
import numpy as np
import argparse
from pathlib import Path
import torchvision.transforms as T
from modules import Net
from helper_funcs import accuracy
import logger
from modules import create_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--n_epochs", default=100, type=int)
    '''optimizer'''
    parser.add_argument("--max_lr", default=3e-4, type=float)
    parser.add_argument('--ema', default=0.995, type=float)
    '''loss'''
    parser.add_argument("--loss_type", default="ce", type=str)
    '''debug'''
    parser.add_argument("--save_path", default=None, type=Path)
    parser.add_argument("--load_path", default=None, type=Path)
        
    args = parser.parse_args()
    return args

def create_dataset(args):
    train_augs = T.Compose([T.ColorJitter(hue=.05, saturation=.05),
                            T.RandomHorizontalFlip(p=0.5),                             
                            T.RandomResizedCrop(size=(32, 32), scale=(0.5, 1.0)),
                            T.ToTensor(), 
                            T.Normalize([0.5]*3, [0.5]*3)])
    
    test_augs = T.Compose([T.ToTensor(), 
                           T.Normalize([0.5]*3, [0.5]*3)])                            
    train_set = Dataset(train=True, transform=train_augs, download=True, root='data')    
    test_set = Dataset(train=False, transform=test_augs, download=True, root='data')
    return train_set, test_set

def train():
    args = parse_args()    
    with args.cfg.open() as f:
         args = yaml.load(f)   
    
    root = Path(args.save_path)
    load_root = Path(args.load_path) if args.load_path else None    
    root.mkdir(parents=True, exist_ok=True)       
    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    writer = SummaryWriter(str(root))

    '''data'''
    train_set, test_set = create_dataset(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    '''net'''
    net = create_net(args)
    net.train()
    net.to(device)    
    '''optimizer'''
    opt = optim.AdamW(net.parameters(), lr=args.max_lr, betas=(0.9, 0.99))    
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                       max_lr=args.max_lr,
                                                       steps_per_epoch=len(train_loader),
                                                       epochs=args.n_epochs,
                                                       pct_start=0.1,
                                                       )
    if args.ema:
        from ema import ModelEma as EMA
        ema = EMA(net, decay_per_epoch=args.ema)
        epochs_from_last_reset = 0
        decay_per_epoch_orig = args.ema

    '''loss'''
    if args.loss_type == "ce":
        loss_cls = nn.CrossEntropyLoss(reduction="sum")
    elif args.loss_type == 'label_smooth':
        from losses import LabelSmoothCrossEntropyLoss
        loss_cls = LabelSmoothCrossEntropyLoss(reduction="sum", smoothing=0.1)
    else:
        raise ValueError("wrong loss, received {}".format(args.loss_type))

    torch.backends.cudnn.benchmark = True
    best_acc = -1
    steps = 0        
    if load_root and load_root.exists():
        checkpoint = torch.load(load_root / "chkpnt.pt")
        net.load_state_dict(checkpoint['model_dict'])
        opt_inner.load_state_dict(checkpoint['opt_dict'])
        steps = checkpoint['resume_step'] if 'resume_step' in checkpoint.keys() else 0
        best_acc = checkpoint['best_acc']
        print('checkpoints loaded')

    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt_inner, max_lr=args['optimizer']['max_lr'], steps_per_epoch=len(train_loader), epochs=args['train']['epochs'])

    for epoch in range(1, args['train']['epochs'] + 1):
        for iterno, (x, y) in enumerate(train_loader):
            net.zero_grad()

            x = x.to(device)            
            y = y.to(device)
            y_est = net(x)                        
            loss = F.cross_entropy(y_est, y)

            loss.backward()            
            opt_inner.step()
            scheduler.step()
            
            acc = accuracy(y_est, target=y, topk=(1,))[0]
            costs.append([loss.item(), acc.item()])
            ######################
            # Update tensorboard #
            ######################             
            writer.add_scalar("train/loss", costs[-1][0], steps)
            writer.add_scalar("train/acc1", costs[-1][1], steps)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], steps)            

            steps += 1            
            acc_test = 0
            loss_test = 0
            if steps % args['logging']['save_interval'] == 0:                
                with torch.no_grad():
                    net.eval()
                    for i, (x, y) in enumerate(test_loader):
                        x = x.to(device)
                        y = y.to(device)
                        y_est = net(x)
                        loss_test += F.cross_entropy(y_est, y).item()
                        acc_test += accuracy(y_est, target=y, topk=(1, ))[0]
                
                net.train()

                loss_test /= len(test_loader)
                acc_test /= len(test_loader)                

                writer.add_scalar("test/loss", loss_test, steps)
                writer.add_scalar("test/acc1", acc_test, steps)
                
                if np.asarray(costs).mean(0)[-1] > best_acc:
                    best_acc = np.asarray(costs).mean(0)[-1]
                    chkpnt = {
                        'model_dict': net.state_dict(),
                        'opt_dict': opt_inner.state_dict(),
                        'step': steps,
                        'best_acc': acc                        
                    }
                    torch.save(chkpnt, root / "chkpnt.pt")
                if steps % args['logging']['log_interval'] == 0:
                    print(
                        "Epoch {} | Iters {} / {} | loss {}".format(
                            epoch,
                            iterno,
                            len(train_loader),                            
                            np.asarray(costs).mean(0),
                        )
                    )
                    costs = []

def train_maml():
    args = parse_args()
    with args.cfg.open() as f:
         args = yaml.load(f)   
    
    root = Path(args['train']['save_path'])
    load_root = Path(args['train']['load_path']) if args['train']['load_path'] else None
    print(load_root)
    root.mkdir(parents=True, exist_ok=True)    
    
    train_augs = T.Compose([T.RandomHorizontalFlip(p=0.5), 
                            T.RandomResizedCrop(size=(32, 32)),                            
                            T.ToTensor(), 
                            T.Normalize([0.5]*3, [0.5]*3)])

    test_augs = T.Compose([T.ToTensor(), 
                           T.Normalize([0.5]*3, [0.5]*3)])
    train_dataset = Dataset(train=True, transform=train_augs, download=True, root='data')
    test_dataset = Dataset(train=False, transform=test_augs, download=True, root='data')

    train_loader = DataLoader(train_dataset, batch_size=args['dataloader']['batch_sz'], shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args['dataloader']['batch_sz'], shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

    net = Net().to(device)    
    opt_inner = optim.Adam(net.parameters(), lr=args['optimizer']['lr'], betas=args['optimizer']['betas'])

    print(net)
    
    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    writer = SummaryWriter(str(root))

    torch.backends.cudnn.benchmark = True
    best_acc = -1
    steps = 0
    acc = 0
    costs = []  
    net.train()    

    iter_train_loader = iter(train_loader)
    iter_test_loader = iter(test_loader)

    for iter_maml in range(10):
        x, y = next(iter_train_loader)
        y_est = net(x)
        inner_loss = F.cross_entropy(y_est, y)
        net.zero_grad()
        grads = torch.autograd.grad(inner_loss, create_graph=True)


    for epoch in range(1, args['train']['epochs'] + 1):
        for iterno, (x, y) in enumerate(train_loader):
            net.zero_grad()

            x = x.to(device)
            y = y.to(device)
            y_est = net(x)
            loss = F.cross_entropy(y_est, y)            
            loss.backward()            
            opt_inner.step()
            
            acc = accuracy(y_est, target=y, topk=(1,))[0]
            costs.append([loss.item(), acc.item()])
            ######################
            # Update tensorboard #
            ######################             
            writer.add_scalar("train/loss", costs[-1][0], steps)
            writer.add_scalar("train/acc1", costs[-1][1], steps)            

            steps += 1            
            acc_test = 0
            loss_test = 0
            if steps % args['logging']['save_interval'] == 0:                
                with torch.no_grad():
                    net.eval()
                    for i, (x, y) in enumerate(test_loader):
                        x = x.to(device)
                        y = y.to(device)
                        y_est = net(x)
                        loss_test += F.cross_entropy(y_est, y).item()
                        acc_test += accuracy(y_est, target=y, topk=(1, ))[0]
                
                net.train()

                loss_test /= len(test_loader)
                acc_test /= len(test_loader)                

                writer.add_scalar("test/loss", costs[-1][0], steps)
                writer.add_scalar("test/acc1", costs[-1][1], steps)
                
                if np.asarray(costs).mean(0)[-1] > best_acc:
                    best_acc = np.asarray(costs).mean(0)[-1]
                    chkpnt = {
                        'model_dict': net.state_dict(),
                        'opt_dict': opt_inner.state_dict(),
                        'step': steps,
                        'best_acc': acc                        
                    }
                    torch.save(chkpnt, root / "chkpnt.pt")
                if steps % args['logging']['log_interval'] == 0:
                    print(
                        "Epoch {} | Iters {} / {} | loss {}".format(
                            epoch,
                            iterno,
                            len(train_loader),                            
                            np.asarray(costs).mean(0),
                        )
                    )
                    costs = []

if __name__ == "__main__":
    train()