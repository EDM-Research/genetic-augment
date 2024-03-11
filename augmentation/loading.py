import time

import torch
import torchvision.transforms as T

from augmentation import COLLECTION, AugmentationSetting
from augmentation.custom import AlbumentationWrapper, PickMultiple


def get_augmentation(name, strength, probability):
    if name == "posterize":
        strength = 1.0 / strength

    parameters = {}
    for param, value in COLLECTION[name]["parameters"].items():
        to_int = False
        if isinstance(value, tuple):
            if isinstance(value[0], int):
                to_int = True
            parameters[param] = (value[0], max(value[0], strength * value[1]))
            if to_int:
                parameters[param] = tuple(map(int, parameters[param]))
                if parameters[param][1] % 2 == 0:
                    parameters[param] = (parameters[param][0], parameters[param][1] + 1)

            if name == "sharpen":
                parameters[param] = (max(0.0, parameters[param][0]), min(parameters[param][1], 1.0))
        else:
            if isinstance(value, int):
                new_value = int(strength * value)
                if new_value % 2 == 0 and name != "posterize":
                    new_value += 1
            else:
                new_value = strength * value
            parameters[param] = new_value

            if name == "zoom_blur":
                parameters[param] = max(new_value, 1.0)

            if name == "unsharp_mask":
                parameters[param] = min(new_value, 1.0)

            if name == "posterize":
                parameters[param] = min(8, new_value)

            if name == "motion_blur":
                parameters[param] = max(3, new_value)

    augmentation = COLLECTION[name]["function"](p=probability, **parameters)

    return AlbumentationWrapper(augmentation)


def get_augmentation_policy(setting: AugmentationSetting):
    augmentation_list = []
    probabilities = []
    for aug_no in range(len(setting.augmentations)):
        if isinstance(setting.augmentations[aug_no], list):
            augmenter = T.Compose([get_augmentation(name, strength, prob) for name, strength, prob in zip(setting.augmentations[aug_no], setting.strengths[aug_no], setting.probabilities[aug_no])])
            probabilities.append(sum(setting.probabilities[aug_no]))
        else:
            if setting.pick_n is not None:
                augmenter = get_augmentation(setting.augmentations[aug_no], setting.strengths[aug_no], 1.0)
                probabilities.append(setting.probabilities[aug_no])
            else:
                augmenter = get_augmentation(setting.augmentations[aug_no], setting.strengths[aug_no], setting.probabilities[aug_no])

        augmentation_list.append(augmenter)

    if setting.pick_n is not None:
        return T.Compose([PickMultiple(augmentation_list, probabilities, k=setting.pick_n), T.ToTensor(), T.ConvertImageDtype(torch.float)])
    else:
        return T.Compose(augmentation_list + [T.ToTensor(), T.ConvertImageDtype(torch.float)])
