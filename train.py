import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from pathlib import Path
from utils.helper_funcs import add_weight_decay
import utils.logger as logger
from metrics import accuracy
from utils.helper_funcs import count_parameters, measure_inference_time
# from clearml import Task

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    '''net'''
    parser.add_argument("--tf_type", default="my", type=str)
    parser.add_argument("--n_classes", default=10, type=int)
    '''optimizer'''
    parser.add_argument("--max_lr", default=3e-4, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument('--ema', default=0.995, type=float)
    parser.add_argument("--amp", action='store_true', default=False)        
    '''debug'''
    parser.add_argument("--save_path", default='outputs/tmp', type=Path)
    parser.add_argument("--load_path", default=None, type=Path)
    parser.add_argument("--save_interval", default=100, type=int)    
    parser.add_argument("--log_interval", default=100, type=int)
    parser.add_argument("--use_fda", default=False, action="store_true")    
    
    args = parser.parse_args()
    return args

def train():
    args = parse_args()

    root = Path(args.save_path)
    load_root = Path(args.load_path) if args.load_path else None    
    root.mkdir(parents=True, exist_ok=True)
    # task = Task.init(project_name='classification', task_name='transformer')
    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    writer = SummaryWriter(str(root))
        
    ####################################
    # Data #
    ####################################
    from datasets.data_utils import create_dataset
    train_set, test_set = create_dataset(args)
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=args.batch_size, 
                                               shuffle=True, drop_last=True, 
                                               num_workers=args.num_workers, 
                                               pin_memory=False)
    
    test_loader = torch.utils.data.DataLoader(test_set, 
                                              batch_size=args.batch_size, 
                                              shuffle=False, 
                                              drop_last=False, 
                                              num_workers=args.num_workers, 
                                              pin_memory=False)
    
    if args.use_fda:
        from datasets.fda import FDA
        fda = FDA(p=0.5, lambda_max=0.1)        

    ####################################
    # Network #
    ####################################
    from modules.models import Net
    net = Net(emb_dim=128, n_classes=args.n_classes, nf=64, tf_type=args.tf_type, factors=[2], inp_sz=(32, 32))        
    net.to(device)
    print("#params={} Mparams".format(count_parameters(net)/1e6))
    t_infer = measure_inference_time(net, torch.randn(1, 3, 32, 32))
    print("time={}+-{} ms".format(t_infer[0], t_infer[1]))
    
    '''loss'''
    from losses import HSIC
    l_ce = nn.CrossEntropyLoss(reduction="sum", label_smoothing=0.1).to(device)
    l_hsic = HSIC(reduction='sum')

    '''optimizer'''
    if args.amp:
        from torch.cuda.amp import GradScaler
        scaler = torch.amp.GradScaler(init_scale=2**10, device=device)
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

    if load_root and load_root.exists():
        checkpoint = torch.load(load_root / "chkpnt.pt")
        net.load_state_dict(checkpoint['model_dict'])
        opt.load_state_dict(checkpoint['opt_dict'])
        steps = checkpoint['resume_step'] if 'resume_step' in checkpoint.keys() else 0
        del checkpoint
        print('checkpoints loaded')        

    
    torch.backends.cudnn.benchmark = True
    acc_test = 0
    steps = 0        
    skip_scheduler = False

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
            
            if args.use_fda:
                x = fda(x)
            
            with torch.amp.autocast(enabled=scaler is not None, device_type=device.type):
                y_est = net(x)                
                loss_cls = l_ce(y_est, y)
                loss_hsic = l_hsic(F.one_hot(y, num_classes=args.n_classes)-y_est.softmax(-1), x.view(args.batch_size, -1))
                loss = loss_cls + loss_hsic
                
            if args.amp:
                scaler.scale(l_ce).backward()
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
            acc = accuracy(y_est, target=y, topk=(1,))[0]
            
            metric_logger.update(loss=loss.item())
            metric_logger.update(hsic=loss_hsic.item())
            metric_logger.update(acc=acc)
            metric_logger.update(lr=opt.param_groups[0]["lr"])
            ######################
            # Update tensorboard #
            ######################               
            steps += 1                        
            if steps % args.save_interval != 0:                                
                writer.add_scalar("lr",lr_scheduler.get_last_lr()[0], steps)
                writer.add_scalar("ce/train", loss.item(), steps)
                writer.add_scalar("hsic/train", loss_hsic.item(), steps)
                writer.add_scalar("acc/train", acc, steps)

            else:
                acc_test = 0
                loss_test = 0                
                net.eval()
                with torch.no_grad():                                        
                    for i, (x, y) in enumerate(test_loader):                                                
                        x = x.to(device)
                        y = y.to(device)
                        y_est = net(x)
                        loss_test += l_ce(y_est, y).item()                        
                        acc_test += accuracy(y_est, target=y, topk=(1, ))[0]
                                
                loss_test /= len(test_loader)
                acc_test /= len(test_loader)                

                writer.add_scalar("ce/test", loss_test, steps)
                writer.add_scalar("acc/test", acc_test, steps)
                
                metric_logger.update(loss_test=loss_test)
                metric_logger.update(acc_test=acc_test)

                net.train()                
                                
                chkpnt = {
                    'model_dict': net.state_dict(),
                    'opt_dict': opt.state_dict(),
                    'step': steps,
                    'best_acc': acc                        
                }
                torch.save(chkpnt, root / "chkpnt.pt")

if __name__ == "__main__":
    train()