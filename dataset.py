import os.path
import random

import numpy as np
import torch.utils.data
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.io import read_image

from transforms import SmallestSideResize


def is_image(file: str) -> bool:
    image_extensions = ["png", "jpg", "jpeg"]

    return file.split(".")[-1].lower() in image_extensions


def get_all_images(root_dir):
    all_files = []

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(root_dir):
        # Append all files in the current directory to the list
        for file in files:
            if is_image(file):
                file_path = os.path.join(root, file)
                all_files.append(file_path)

    return all_files


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path: str, transforms = None, in_memory=True, max_images=None):
        self.dataset = folder_path

        self.image_paths = get_all_images(folder_path)
        random.shuffle(self.image_paths)

        self.transforms = transforms
        self.resize = SmallestSideResize(600)
        self.default_transforms = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float)
        ])

        self.in_memory = in_memory
        self.images = None
        if in_memory:
            self.images = []
            max_images = min(len(self.image_paths), max_images) if max_images is not None else len(self.image_paths)
            for i in range(max_images):
                self.images.append(self.load_image(i))

    def __len__(self):
        return len(self.dataset)

    def load_image(self, index: int) -> np.array:
        image = read_image(self.image_paths[index]).permute(1, 2, 0).numpy()

        if len(image.shape) == 3 and image.shape[2] == 1:
            # Convert grayscale image to RGB
            image = np.repeat(image, 3, axis=2)

        return image

    def __getitem__(self, item):
        if self.in_memory:
            image = self.images[item]
        else:
            image = self.load_image(item)

        if self.transforms:
            image = self.transforms(image)
        else:
            image = self.default_transforms(image)

        image, target = self.resize(image, {"boxes": []})
        image = torch.clamp(image, 0.0, 1.0)

        return image

    @staticmethod
    def collate_fn(batch):
        images = batch

        height = max([image.size(1) for image in images])
        width = max([image.size(2) for image in images])

        padded_images = []

        for image in images:
            h_diff = (height - image.size(1))
            wdiff = (width - image.size(2))
            hl_padding = h_diff // 2
            wl_padding = wdiff // 2

            hr_padding = h_diff - hl_padding
            wr_padding = wdiff - wl_padding

            image = F.pad(image, [int(wl_padding), int(hl_padding), int(wr_padding), int(hr_padding)])

            padded_images.append(image)

        return padded_images
