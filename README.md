# resnet-numpy-cifar10

Pure numpy implementation of a basic ResNet. Evaluation of implementation on Cifar-10. Pure numpy implementation allows, e.g., integration into JAX. Implementation was created in context of homework in a Master's course at the Technical University of Munich and reached 7th place in a leaderboard among >500 homework submissions.

## Setup

0. Clone this repository. Note: Dataset is managed with `git lfs` and pulled together with the code when cloning this repository.
1. Create a fresh python virtual environment with python version 3.10.13 and source environment.
2. Setup environment with _poetry_

```s
$ pip install --upgrade pip
$ pip install poetry
$ poetry install
```

## Dataset

This repository uses `git lfs` (large file storage) for the cifar-10 dataset. If the dataset was not pulled upon cloning, follow following steps:

1. Make sure `git lfs` is installed on your system. On manjaro, e.g., `git lfs` can be installed via

```s
$ sudo pacman -Syu git-lfs
```

2. Pull the dataset via `git lfs`

```s
$ git lfs ls-files
$ git lfs pull
```

## Usage

Training script will train a basic ResNet (one conv layer, one residual block, one affine layer) on cifar-10 for 10 epochs. At the end of the script, the accuracies on training, test and validation set are displayed.

```s
$ python train.py
```

### Reference output

The output should look like:

```
(Epoch 1 / 10) train loss: 2.301953; val loss: 2.302474
(Epoch 2 / 10) train loss: 1.709292; val loss: 1.549400
(Epoch 3 / 10) train loss: 1.434347; val loss: 1.415999
(Epoch 4 / 10) train loss: 1.344497; val loss: 1.408266
(Epoch 5 / 10) train loss: 1.266230; val loss: 1.329670
(Epoch 6 / 10) train loss: 1.208818; val loss: 1.272511
(Epoch 7 / 10) train loss: 1.153916; val loss: 1.265599
(Epoch 8 / 10) train loss: 1.118229; val loss: 1.231749
(Epoch 9 / 10) train loss: 1.082194; val loss: 1.225092
(Epoch 10 / 10) train loss: 1.054310; val loss: 1.247250
Training accuray: 0.63901
Validation set accuray: 0.57126
Test set accuray: 0.57121
```