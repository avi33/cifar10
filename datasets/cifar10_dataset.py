import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

class CustomCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, pairs=False):
        # Call the super constructor to initialize the dataset
        super().__init__(root, train, transform, target_transform, download)
        self.pairs = pairs

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if not self.pairs:
            # Convert image to tensor and apply transformations
            img = self.transform(img)

            # Return the image, target, and index as a tuple
            return img, target
        
        else:
            # Convert image to tensor and apply transformations
            img_orig = Image.fromarray(img.copy())
            img = self.transform(img_orig)
            imga = self.transform(img_orig)

            # Return the image, target, and index as a tuple
            return img, target, imga
        