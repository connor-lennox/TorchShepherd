from typing import Mapping, Callable, Type, List

import torch
from torch import optim
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

import train_utils


class Trainer:
    """A default model trainer.

    Provided with a builder function, loss function, optimizer builder, and optional forward function, 
    will train a model over a provided PyTorch dataset (additional training parameters also available).
    """
    def __init__(self, model_builder: Callable[[], nn.Module] = None, loss_func: Callable[[torch.Tensor,  torch.Tensor], torch.Tensor] = None,
                    optim_builder: Callable[[nn.Module], Optimizer] = None, alt_forward: Callable[[nn.Module, torch.Tensor], torch.Tensor] = None,
                    device: str = 'cpu') -> None:
        self.model_builder: Callable[[], nn.Module] = model_builder
        self.loss_func: Callable[[torch.Tensor,  torch.Tensor], torch.Tensor] = loss_func
        self.optim_builder: Callable[[nn.Module], Optimizer] = optim_builder
        self.alt_forward: Callable[[nn.Module, torch.Tensor], torch.Tensor] = alt_forward

        self.loader_builder: Callable[[Dataset], DataLoader] = None

        self.device: str = device

    def train(self, data: Dataset, epochs: int) -> nn.Module:
        # Construct model via builder function and send to Trainer device
        model: nn.Module = self.model_builder().to(self.device)
        optimizer: Optimizer = self.optim_builder(model)

        # Construct custom or default loader
        dataloader: DataLoader = self.loader_builder(data) if self.loader_builder is not None else DataLoader(data)

        # Actual training loop
        self._do_train(model, optimizer, dataloader, epochs)

        return model

    def cross_validation_train(self, data: List[Dataset], epochs: int) -> List[nn.Module]:
        """Performs cross validation training.

        For each dataset in the list, a model will be trained on all datasets *except* that one.
        Models are returned back as a List, with the index representing the dataset the model 
        was *not* trained with. For example, the model at index 0 was trained on datasets [1, 2, 3, ...],
        but was not trained on dataset 0.
        """
        models = []

        for model_index in range(len(data)):
            model: nn.Module = self.model_builder().to(self.device)
            optimizer: Optimizer = self.optim_builder(model)

            dataloader: DataLoader = torch.utils.data.ChainDataset(data[:model_index] + data[model_index+1:])

            self._do_train(model, optimizer, dataloader, epochs)

            models.append(model)

        return models

    def _do_train(self, model: nn.Module, optimizer: Optimizer, dataloader: DataLoader, epochs: int) -> None:
        for epoch in range(epochs):
            for batch_idx, samples in enumerate(dataloader):
                optimizer.zero_grad()

                # Create output from the model and the xs of the sample, and calculate loss using ys of sample
                output = self.alt_forward(model, samples[0]) if self.alt_forward is not None else model(samples)
                loss = self.loss_func(output, samples[1])

                loss.backward()
                optimizer.step()