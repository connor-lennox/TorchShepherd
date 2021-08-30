from typing import Callable, List

import torch
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
                 loader_builder: Callable[[Dataset], DataLoader] = None, device: str = 'cpu') -> None:

        self.model_builder: Callable[[], nn.Module] = model_builder
        self.loss_func: Callable[[torch.Tensor,  torch.Tensor], torch.Tensor] = loss_func
        self.optim_builder: Callable[[nn.Module], Optimizer] = optim_builder
        self.alt_forward: Callable[[nn.Module, torch.Tensor], torch.Tensor] = alt_forward

        self.loader_builder: Callable[[Dataset], DataLoader] = loader_builder

        self.device: str = device

    def train(self, data: Dataset, epochs: int, verbose: bool = False) -> nn.Module:
        # Construct model via builder function and send to Trainer device
        model: nn.Module = self.model_builder().to(self.device)
        optimizer: Optimizer = self.optim_builder(model)

        # Construct custom or default loader
        dataloader: DataLoader = self.loader_builder(data) if self.loader_builder is not None else DataLoader(data)

        # Actual training loop
        self._do_train(model, optimizer, dataloader, epochs, verbose=verbose)

        return model

    def test(self, model: nn.Module, data: Dataset, verbose: bool = False):
        dataloader: DataLoader = self.loader_builder(data) if self.loader_builder is not None else DataLoader(data)
        self._do_test(model, dataloader, verbose=verbose)

    def cross_validation_train(self, data: List[Dataset], epochs: int, verbose: bool = False) -> List[nn.Module]:
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

            comb_dataset: Dataset = torch.utils.data.ChainDataset(data[:model_index] + data[model_index+1:])
            dataloader: DataLoader = self.loader_builder(comb_dataset) if self.loader_builder is not None else DataLoader(comb_dataset)

            self._do_train(model, optimizer, dataloader, epochs, verbose=verbose)

            models.append(model)

        return models

    def _do_train(self, model: nn.Module, optimizer: Optimizer, dataloader: DataLoader, epochs: int, verbose: bool = False) -> None:
        model.train()

        num_batches = len(dataloader)
        if verbose:
            print()

        for epoch in range(epochs):
            loss_sum = 0
            for batch_idx, (data, target) in enumerate(dataloader):
                if verbose:
                    print(f"\rEpoch {epoch+1}: {train_utils.progress_string(batch_idx, num_batches)}", end='')

                # Zero optimizer gradient prior to training step
                optimizer.zero_grad()

                # Move tensors to appropriate training device
                data = data.to(self.device)
                target = target.to(self.device)

                # Create output from the model and the xs of the sample, and calculate loss using ys of sample
                output = self.alt_forward(model, data) if self.alt_forward is not None else model(data)
                loss = self.loss_func(output, target)

                # Track total loss over epoch
                loss_sum += loss.item()

                # Perform optimization step
                loss.backward()
                optimizer.step()
            
            if verbose:
                print(f"\rEpoch {epoch+1}: Loss={loss_sum / num_batches}")

    def _do_test(self, model: nn.Module, dataloader: DataLoader, verbose: bool = False):
        model.eval()

        num_batches = len(dataloader)
        if verbose:
            print()

        loss_sum = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            if verbose:
                print(f"\rTesting: {train_utils.progress_string(batch_idx, num_batches)}", end='')

            # Move tensors to appropriate testing device
            data = data.to(self.device)
            target = target.to(self.device)

            # Create output from the model and the xs of the sample, and calculate loss using ys of sample
            output = self.alt_forward(model, data) if self.alt_forward is not None else model(data)
            loss = self.loss_func(output, target)

            # Track total loss over test
            loss_sum += loss.item()

        if verbose:
            print(f"\rTesting: Loss={loss_sum / num_batches}")
