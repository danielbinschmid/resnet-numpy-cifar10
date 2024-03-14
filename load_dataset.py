import matplotlib.pyplot as plt
import numpy as np
import os

from codebase.data import (
    DataLoader,
    ImageFolderDataset,
    RescaleTransform,
    NormalizeTransform,
    ReshapeTransform,
    FlattenTransform,
    ComposeTransform,
)
from codebase.data.image_folder_dataset import RandomHorizontalFlip

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# ------------------------------------

def get_dataloaders(datasets, batch_size=256):
    dataloaders = {}

    for mode in ['train', 'val', 'test']:
        crt_dataloader = DataLoader(
            dataset=datasets[mode],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        dataloaders[mode] = crt_dataloader
    return dataloaders

def get_compose_transform(useFlatten=True, training=False):

    # Use the Cifar10 mean and standard deviation computed in Exercise 3.
    cifar_mean = np.array([0.5, 0.5, 0.5]) # np.array([0.49191375, 0.48235852, 0.44673872])
    cifar_std  = np.array([0.5, 0.5, 0.5]) # np.array([0.24706447, 0.24346213, 0.26147554])

    rescale_transform = RescaleTransform()
    normalize_transform = NormalizeTransform(
        mean=cifar_mean,
        std=cifar_std
    )

    reshape_transform = ReshapeTransform()
    flip = RandomHorizontalFlip()
    if not training:
        compose_transform = ComposeTransform([rescale_transform, 
                                                normalize_transform,
                                                reshape_transform])
    else:
        compose_transform = ComposeTransform([flip,
                                            rescale_transform, 
                                                normalize_transform,
                                                reshape_transform])
    return compose_transform

def get_datasets(DATASET, cifar_root, compose_transform, compose_transform_training=None):
    # Create a train, validation and test dataset.
    datasets = {}
    for mode in ['train', 'val', 'test']:
        if compose_transform_training is not None and mode == 'train':
            crt_dataset = DATASET(
                mode=mode,
                root=cifar_root, 
                transform=compose_transform_training,
                split={'train': 0.65, 'val': 0.15, 'test': 0.2}
            )
        else:
            crt_dataset = DATASET(
                mode=mode,
                root=cifar_root, 
                transform=compose_transform,
                split={'train': 0.65, 'val': 0.15, 'test': 0.2}
            )
        datasets[mode] = crt_dataset
    return datasets
