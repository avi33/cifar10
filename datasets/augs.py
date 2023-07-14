import numpy as np
import torchvision.transforms as T
import random
from PIL import Image

class RandomColor:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, sample):
        if random.random() < self.p:
            idx = np.random.permutation(3)
            sample = Image.fromarray(np.array(sample)[:, :, idx])
        return sample
    
class RandomResizeCrop:
    def __init__(self, scales=(0.8, 1.2), p=0.5) -> None:
        self.p = p
        self.scales = scales

    def __call__(self, image):
        # Convert PIL image to NumPy array
        image = np.array(image)

        # Extract the original size from the input image
        original_size = image.shape[:2]

        # Generate a random scale between 0.8 and 1.2
        scale = np.random.uniform(self.scales[0], self.scales[1])

        # Perform resize and crop operations based on the scale
        if scale < 1:
            # Resize and pad to the original size
            new_size = tuple(int(dim * scale) for dim in original_size)
            resized_image = Image.fromarray(image)
            if len(image.shape) == 2:
                resized_image = resized_image.convert('L')  # Convert to grayscale
            resized_image = resized_image.resize(new_size, Image.BILINEAR)
            padded_image = Image.new(resized_image.mode, original_size)
            padded_image.paste(resized_image, ((original_size[0] - new_size[0]) // 2, (original_size[1] - new_size[1]) // 2))
            return padded_image
        else:
            # Resize and crop to the original size
            new_size = tuple(int(dim * scale) for dim in original_size)
            resized_image = Image.fromarray(image)
            if len(image.shape) == 2:
                resized_image = resized_image.convert('L')  # Convert to grayscale
            resized_image = resized_image.resize(new_size, Image.BILINEAR)
            left = (new_size[0] - original_size[0]) // 2
            top = (new_size[1] - original_size[1]) // 2
            cropped_image = resized_image.crop((left, top, left + original_size[0], top + original_size[1]))
            return cropped_image