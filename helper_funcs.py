import torch

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    with torch.no_grad():
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()    
        correct = pred.eq(target.view(1, -1).expand_as(pred))    
    return [correct[:k].view(-1).float().sum(0) * 100. / batch_size for k in topk]

