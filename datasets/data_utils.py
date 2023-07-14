import torchvision.transforms as T

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