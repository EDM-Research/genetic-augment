import random

import albumentations as A
import numpy as np
import torch
import torchvision.transforms.functional as F


class UniformNoise(A.BasicTransform):
    def __init__(self, strength: float = 0.5, p: float = 0.5):
        super().__init__()

        self.strength = strength
        self.p = p

    def __call__(self, image: np.array):
        if random.uniform(0.0, 1.0) > self.p:
            return image

        max_val = 255 if image.dtype == np.uint8 else 1.0

        result = image
        probability = np.random.uniform(0.0, 1.0, size=result.shape)
        swap = probability < self.strength
        noise = np.random.uniform(0, max_val, size=result.shape)
        result[swap] = noise[swap]

        return {
            "image": result
        }


class AlbumentationWrapper(torch.nn.Module):
    """
    Wrapper that applies augmentation to data but passes through label
    WARNING: Only use with augmentations that do not change label (e.g. no image transformation)
    """
    def __init__(self, augmentation: A.BasicTransform):
        super().__init__()
        self.augmentation = augmentation

    def forward(self, image):
        result = self.augmentation(image=image)
        if not isinstance(result, dict):
            return result
        return result["image"]


class PickMultiple(torch.nn.Module):
    def __init__(self, transforms, probabilities, k=1):
        super().__init__()
        self.transforms = transforms
        self.probabilities = probabilities
        self.k = k

    def __call__(self, image):
        selected = random.choices(self.transforms, weights=self.probabilities, k=self.k)

        for t in selected:
            image = t(image)
        return image
