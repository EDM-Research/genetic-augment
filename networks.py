import numpy as np
import torch
import torch.utils.data
import torchvision.transforms.functional as F
from torchvision.models import ResNet18_Weights, resnet18


def get_backbone():
    weights = ResNet18_Weights
    model = resnet18(weights=weights)

    model.fc = torch.nn.Identity()
    model.eval()

    return model


@torch.no_grad()
def compute_features(network: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str, transform: bool = True):
    collected_features = []
    for batch in dataloader:
        images = torch.stack(batch).to(device)
        if transform:
            images = F.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        feature = network(images)
        collected_features.extend(feature.cpu().numpy())

    np_features = np.stack(collected_features)

    return np_features
