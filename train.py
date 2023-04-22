import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import ToPILImage
import yaml
import numpy as np
import argparse
from pathlib import Path
import torchvision.transforms as T
from helper_funcs import add_weight_decay
import logger
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--dataset", default="cifar10", type=str)
    '''net'''
    parser.add_argument("--tf_type", default="my", type=str)
    parser.add_argument("--n_classes", default=10, type=int)
    '''optimizer'''
    parser.add_argument("--max_lr", default=3e-4, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument('--ema', default=0.995, type=float)
    parser.add_argument("--amp", action='store_true', default=False)
    '''loss'''
    parser.add_argument("--loss_type", default="label_smooth", type=str)
    '''debug'''
    parser.add_argument("--save_path", default='outputs/tmp', type=Path)
    parser.add_argument("--load_path", default=None, type=Path)
    parser.add_argument("--save_interval", default=100, type=int)    
    parser.add_argument("--log_interval", default=100, type=int)          
    
    args = parser.parse_args()
    return args

def create_dataset(args):
    if args.dataset == 'cifar10':
        from torchvision.datasets import CIFAR10 as Dataset
        train_augs = T.Compose([#T.ColorJitter(hue=.05, saturation=.05),
                                T.RandomHorizontalFlip(p=0.5),
                                T.RandomRotation(degrees=15),
                                T.RandomResizedCrop(size=(32, 32), scale=(0.5, 1.0)),                            
                                T.ToTensor(), 
                                T.Normalize([0.5]*3, [0.5]*3)])
        
        test_augs = T.Compose([T.ToTensor(), 
                            T.Normalize([0.5]*3, [0.5]*3)])

        train_set = Dataset(train=True, transform=train_augs, download=False, root='data')    
        test_set = Dataset(train=False, transform=test_augs, download=False, root='data')

    elif args.dataset == 'svhn':
        from torchvision.datasets import SVHN as Dataset
        train_augs = T.Compose([#T.ColorJitter(hue=.05, saturation=.05),
                                T.RandomHorizontalFlip(p=0.5),
                                T.RandomRotation(degrees=15),
                                T.RandomResizedCrop(size=(32, 32), scale=(0.5, 1.0)),                            
                                T.ToTensor(), 
                                T.Normalize([0.5]*3, [0.5]*3)])
        
        test_augs = T.Compose([T.ToTensor(), 
                            T.Normalize([0.5]*3, [0.5]*3)])

        train_set = Dataset(train=True, transform=train_augs, download=False, root='data')    
        test_set = Dataset(train=False, transform=test_augs, download=False, root='data')

    elif args.dataset == 'tinyimagenet':
        from data import TinyImageNetDataset as Dataset
        train_set = Dataset(train=True, transform=train_augs, download=False, root='data')    
        test_set = Dataset(train=False, transform=test_augs, download=False, root='data')
    
    else:
        raise ValueError("wrong dataset {}".format(args.dataset))
    
    return train_set, test_set

def train():
    args = parse_args()

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
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    '''net'''
    from modules.models import Net
    net = Net(emb_dim=128, n_classes=args.n_classes, nf=16, tf_type=args.tf_type, factors=[2, 2, 2], inp_sz=(32, 32))
    # if int(torch.__version__.split('.')[0]) > 1:
    #     net = torch.compile(net)
    #     print("compiled")
    #     torch._dynamo.config.suppress_errors = True
    # from RepVGG.repvggplus import create_RepVGGplus_by_name
    # net = create_RepVGGplus_by_name("RepVGG-A1", deploy=False, use_checkpoint=False)
    # net = Net(emb_dim=128, n_classes=args.n_classes, nf=16, tf_type=args.tf_type)
    # from modules.fftlayer import Net
    # net = Net(nf=16)
    # net.Linear = nn.Linear(1280, args.n_classes)
    net.to(device)
    
    '''optimizer'''
    if args.amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler(init_scale=2**10)
        eps = 1e-4
    else:
        scaler = None
        eps = 1e-8
    
    parameters = add_weight_decay(net, weight_decay=args.wd, skip_list=())

    opt = optim.AdamW(parameters, lr=args.max_lr, betas=(0.9, 0.99), eps=eps, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                       max_lr=args.max_lr,
                                                       steps_per_epoch=len(train_loader),
                                                       epochs=args.n_epochs,
                                                       pct_start=0.1,                                                       
                                                    )    
    if args.ema is not None:
        from ema import ModelEma as EMA
        ema = EMA(net, decay_per_epoch=args.ema)
        epochs_from_last_reset = 0
        decay_per_epoch_orig = args.ema

    '''loss'''
    if args.loss_type == "ce":
        criterion = nn.CrossEntropyLoss(reduction="sum").to(device)
    elif args.loss_type == 'label_smooth':
        from losses import LabelSmoothCrossEntropyLoss
        criterion = LabelSmoothCrossEntropyLoss(reduction="sum", smoothing=0.1).to(device)
    elif args.loss_type == 'label_smooth_dirichlet':
        from modules.dirichlet import EstDirichlet        
        D = EstDirichlet(n_classes=10, n_iters=100, tol=1e-3)
        from losses import LabelSmoothDirichletCrossEntropyLoss
        # criterion = LabelSmoothCrossEntropyLoss(reduction="sum", smoothing=0.1).to(device)
        criterion = LabelSmoothDirichletCrossEntropyLoss(reduction="sum").to(device)      
        # criterion = nn.NLLLoss(reduction="sum").to(device)
    else:
        raise ValueError("wrong loss, received {}".format(args.loss_type))
    from losses import HSIC
    H = HSIC(reduction='sum')    
    loss_ce = nn.CrossEntropyLoss(reduction="sum").to(device)    
    
    torch.backends.cudnn.benchmark = True
    acc_test = 0
    steps = 0        
    skip_scheduler = False    

    if load_root and load_root.exists():
        checkpoint = torch.load(load_root / "chkpnt.pt")
        net.load_state_dict(checkpoint['model_dict'])
        opt.load_state_dict(checkpoint['opt_dict'])
        steps = checkpoint['resume_step'] if 'resume_step' in checkpoint.keys() else 0
        print('checkpoints loaded')

    for epoch in range(1, args.n_epochs + 1):
        metric_logger = logger.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", logger.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"
        if args.ema is not None:
            if epochs_from_last_reset <= 1:  # two first epochs do ultra short-term ema
                ema.decay_per_epoch = 0.01
            else:
                ema.decay_per_epoch = decay_per_epoch_orig
            epochs_from_last_reset += 1
            # set 'decay_per_step' for the eooch
            ema.set_decay_per_step(len(train_loader))        
        for iterno, (x, y) in  enumerate(metric_logger.log_every(train_loader, args.log_interval, header)):        
            net.zero_grad(set_to_none=True)
            x = x.to(device)            
            y = y.to(device)
            with torch.cuda.amp.autocast(enabled=scaler is not None):                
                y_est = net(x)
                loss_cls = criterion(y_est, y)
                loss_hsic = H(F.one_hot(y, num_classes=args.n_classes)-y_est, x.view(args.batch_size, -1))
                loss = loss_cls + loss_hsic

            if args.amp:
                scaler.scale(criterion).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
                scaler.step(opt)
                amp_scale = scaler.get_scale()
                scaler.update()
                skip_scheduler = amp_scale != scaler.get_scale()
            else:
                loss.backward()
                opt.step()

            if args.ema is not None:
                ema.update(net, steps)

            if not skip_scheduler:
                lr_scheduler.step()

            '''metrics'''            
            acc = logger.accuracy(y_est, target=y, topk=(1,))[0]
            
            metric_logger.update(loss=loss.item())
            metric_logger.update(hsic=loss_hsic.item())
            metric_logger.update(acc=acc)
            metric_logger.update(lr=opt.param_groups[0]["lr"])
            ######################
            # Update tensorboard #
            ######################               
            steps += 1                        
            if steps % args.save_interval != 0:
                writer.add_scalar(f"loss/train", loss.item(), steps)
                writer.add_scalar(f"hsic/train", loss_hsic.item(), steps)
                writer.add_scalar(f"acc/train", acc, steps)
                writer.add_scalar(f"lr", lr_scheduler.get_last_lr()[0], steps)
            else:
                acc_test = 0
                loss_test = 0                
                net.eval()
                with torch.no_grad():                                        
                    for i, (x, y) in enumerate(test_loader):                                                
                        x = x.to(device)
                        y = y.to(device)
                        y_est = net(x)
                        loss_test += loss_ce(y_est, y).item()                        
                        acc_test += logger.accuracy(y_est, target=y, topk=(1, ))[0]
                                
                loss_test /= len(test_loader)
                acc_test /= len(test_loader)                

                writer.add_scalar("loss/test", loss_test, steps)
                writer.add_scalar("acc/test", acc_test, steps)

                metric_logger.update(loss_test=loss_test)
                metric_logger.update(acc_test=acc)

                net.train()                
                                
                best_acc = metric_logger.meters['acc_test'].deque[0]
                chkpnt = {
                    'model_dict': net.state_dict(),
                    'opt_dict': opt.state_dict(),
                    'step': steps,
                    'best_acc': acc                        
                }
                torch.save(chkpnt, root / "chkpnt.pt")

if __name__ == "__main__":
    train()