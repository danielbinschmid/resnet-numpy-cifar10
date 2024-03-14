import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import os
from codebase.data import (
    ImageFolderDataset,
)
from codebase.networks import (
    CrossEntropyFromLogits
)

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
from codebase.solver import Solver
from codebase.networks.optimizer import Adam
from codebase.networks import BasicResNet
from load_dataset import get_compose_transform, get_datasets, get_dataloaders

DATASET = ImageFolderDataset
data_zip_file = os.path.abspath(os.path.join(os.curdir, "data", "cifar10.zip"))
cifar_root = os.path.abspath(os.path.join(os.curdir, "data"))
compose_transform = get_compose_transform(useFlatten=False)
compose_transform_training = get_compose_transform(training=True)
datasets = get_datasets(DATASET, cifar_root, compose_transform, compose_transform_training)
dataloaders = get_dataloaders(datasets)

print("unzipping data..")
debugging_validation_dataset = DATASET(
    mode='val',
    root=cifar_root,
    data_zip_file=data_zip_file,
    transform=compose_transform,
    limit_files=100
)
print("done.")

def train_full():
    epochs = 10
    model =  BasicResNet()
    train_loader = dataloaders['train']

    loss = CrossEntropyFromLogits()
    solver = Solver(model, train_loader, dataloaders['val'], 
                    learning_rate=1e-3, loss_func=loss, optimizer=Adam)
    solver.train(epochs=epochs, patience=4)

    print("Training accuray: %.5f" % (solver.get_dataset_accuracy(train_loader)))
    print("Validation set accuray: %.5f" % (solver.get_dataset_accuracy(dataloaders['val'])))
    print("Test set accuray: %.5f" % (solver.get_dataset_accuracy(dataloaders['test'])))

if __name__ == "__main__":
    train_full()