# GeneticAugment for designing sim-to-real augmentation

This repository contains the implementation of the GeneticAugment algorithm.
The algorithm is designed to automatically find good data augmentation strategies for training on synthetic data.
Given an unlabeled synthetic dataset and unlabeled real dataset the algorithm leverages genetic learning to find image augmentations to use on the synthetic data to improve generalization on real data.
The genetic learning is steered by two metrics that increase the variation in the augmented synthetic images and decrease the distance between the augmented images and the real images.

## Install

To install the needed libraries run:
```
pip install -r requirements.txt
```

This repository mainly uses PyTorch for neural networks, Albumentations for augmentations and Deap for genetic learning.

## Usage

To find an augmentation strategy for a given synthetic and real dataset run the following command:
```
python learn.py --train-folder      PATH    # specify path to the folder with synthetic images
                --reference-folder  PATH    # specify path to the folder with real images
```
Images within the folder and all sub-folders are considered.

By default, a sequential strategy is learned without a fixed length. 
The following arguments can be used to change what strategy is learned:
- `--pick-n N` instead of executing all augmentations sequentially  `N` augmentations are chosen at random for execution.
- `--nested-size N` by default all augmentation units are a single augmentation. With this parameter an augmentation unit can consist of multiple single augmentations.
- `--fixed-length N` by default the algorithm is free to learn augmentations of all lengths between `[2, 17]`. With this parameter this can be fixed to length `N`.

For example with `--pick-n 3 --nested-size 2 --fixed-length 5` augmentation strategies will consist of five augmentation units that each have two augmentations.
An image will be augmented by picking three of these double augmentations at random and applying them.

The settings of the genetic learning can be set as follows.
- `--generations N` number of generations to train for. Default is five.
- `--population N` size of the population. Default 100.

The training evolution and pareto frontier are plotted after training.

After training the found policy is printed and saved to `augmentation.pkl` as an `AugmentationSettings` object.
The function `load_augmentation_policy` can be used to transform this settings object into an augmentation pipeline that can be used for training an object detection model.

## Example

In the paper we found augmentation policies for training on the Sim10k dataset and testing on the Cityscapes dataset.
The best performing policy picked two single augmentations from a list with no fixed length.
This policy was learned as follows:
```
python train.py --train-folder      data/sim10k/VOC2012/JPEGImages
                --reference-folder  data/cityscapes/leftImg8bit/train
                --pick-n            2
                --nested-size       1
                --generations       5
                --population        100
```

To make the implementation as general as possible, all images from a folder are selected.
Please note that for the experiments in the paper we ensured that only images from the training set were used for finding the augmentation policy.