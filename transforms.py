import torch
import torchvision.transforms.functional as F


class SmallestSideResize(torch.nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, image, target):
        resized = F.resize(image, size=self.size)

        if len(target["boxes"]) > 0:
            x_scale = resized.size(-2) / image.size(-2)
            y_scale = resized.size(-1) / image.size(-1)
            boxes = [[b[0] * x_scale, b[1] * y_scale, b[2] * x_scale, b[3] * y_scale] for b in target["boxes"]]
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["masks"] = F.resize(target["masks"], size=self.size)

        return resized, target