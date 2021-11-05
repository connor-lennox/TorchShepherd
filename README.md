# TorchShepherd
A PyTorch training harness designed to not reduce the capabilities of training procedures due to overencompassing abstractions. To accomplish this, heavy usages of lambdas allow injecting code directly into Shepherd functions, making the generalized training harness fully customizable to fit your needs.

# Usage
Import the package as `torchshepherd`, and specifically import the `trainers` file:

```py
from torchshepherd import trainers
```

This file has the main class responsible for training/testing PyTorch models.


## Trainer Setup
The `Trainer` class takes several functions as parameters, which allows fine-grain control over the training process while also abstracting away the boiler-plate code.

The required functions are:

- `model_builder`: a function that takes no parameters and returns a `torch.nn.Module`. This is used by the `Trainer` to create an instance of your model, so do any initialization needed in this function.
- `loss_func`: a function taking in two `torch.Tensor` objects and returning a `torch.Tensor`. During training time, this function is provided parameters in the order of `(model output, target output)` so please ensure your loss function matches this schema (built-in PyTorch loss functions in the `torch.nn.functional` ARE valid inputs here: they follow this input format).
- `optim_builder`: a function taking in a `torch.nn.Module` (your model) and producing a `torch.optim.Optimizer`. Generally this looks something like this:

```py
def optimbuilder(model):
  return torch.optim.Adam(model.parameters, lr=0.0001)
```
This function (like all others) can also be provided as a lambda function, which allows it to be in-lined in the `Trainer` constructor.


In addition to these required functions, the following functions are *optional*. If they are not provided, the `Trainer` will fall back to default behavior.

- `alt_forward`: a function taking a `torch.nn.Module` (your model) and a `torch.Tensor` (the input). This function needs to return a `torch.Tensor` representing the output of the model, in a format ready to pass to the loss function. By default, the `__call__` function of the model, which will (usually) call the `forward` function.
- `loader_builder`: a function taking a `torch.utils.data.Dataset` and returning a `torch.utils.data.DataLoader`. This is used internally to prep the data for passing to the model in an efficient manner. If you're using a CUDA enabled training device, you'll want to use this function to setup data pinning and thread workers. By default, a `DataLoader` will be constructed by passing the `Dataset` and no additional parameters.


## Training Procedure
To train a model, call the `train` function on the `Trainer` object. This function takes a `Dataset`, an `int` (epochs) and an optional `bool` (verbosity) parameter. A `DataLoader` will be constructed from the provided data, as outlined above, and a model will be constructed. The model will be trained using this data and returned. If the `verbose` parameter is set to `True`, some information will be printed to stdout during execution.

### Cross-Validation Training
As an alternative to the training process, the `cross_validation_train` function takes in a `List[Dataset]` instead of a single `Dataset`. The result of this function is a `List[torch.nn.Module]`, where each returned model has **not** seen the dataset in it's corresponding index (that is, the model at index 0 will have seen datasets 1, 2, 3, but not dataset 0).

## Testing Procedure
The `test` function of the `Trainer` takes in a model, a dataset, and an optional verbosity `bool`. It will perform a test over the dataset, using the loss function as a metric.


# Todo
While this package currently provides a clean training platform, allowing access to the inner workings of PyTorch while also removing the need to write boiler-plate code, there are a few features it's missing. In no particular order, here are the things I plan on adding at some point in the future:

- Built-in support for PackedSequences, which will allow for much faster training of recurrent models.
- More extensive testing functionality, including the ability to provide additional test metrics (beyond just loss functions).
- An efficient way to serialize model parameters after training, especially with collections of models trained via cross-validation.
- A one-off function that can be used to test a group of cross-validation models.
