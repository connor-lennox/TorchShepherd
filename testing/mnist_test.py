from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchshepherd import trainers


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_mnist_model():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        nn.Flatten(1),
        nn.Linear(9216, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 10),
        nn.LogSoftmax(dim=1)
    )


# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('../data', train=False, download=True, transform=transform)


if DEVICE == 'cuda':
    def construct_loader(dataset):
        return DataLoader(dataset, pin_memory=False, batch_size=128, shuffle=True, num_workers=2)
else:
    def construct_loader(dataset):
        return DataLoader(dataset, batch_size=128, shuffle=True)


trainer = trainers.Trainer(model_builder=make_mnist_model, loss_func=F.nll_loss,
                           optim_builder=lambda m: torch.optim.Adadelta(m.parameters(), lr=0.001),
                           loader_builder=construct_loader, device=DEVICE)

trained_model = trainer.train(train_data, 10, verbose=True)
trainer.test(trained_model, test_data, verbose=True)
