# DropGrad: A Simple Method for Regularization and Accelerated Optimization of Neural Networks

DropGrad is a regularization method for neural networks that works by randomly (and independently) setting gradient values to zero before an optimization step. Similarly to Dropout, it has a single parameter, `drop_rate`, the probability of setting each parameter gradient to zero. In order to de-bias the remaining gradient values, they are divided by `1.0 - drop_rate`.

## Features

- Simple and easy-to-use gradient regularization technique
- Compatible with various optimizers and learning rate schedulers
- Supports per-parameter drop rates for fine-grained control
- Implements drop rate schedulers for dynamic regularization
- Provides an option to apply "full" update drop for further regularization

## Code Structure

```
dropgrad/
│
├── docs/
│   └── analysis.md
│
├── dropgrad/
│   ├── __init__.py
│   ├── dropgrad_opt.py
│   └── dropgrad_scheduler.py
│
├── examples/
│   ├── basic_usage.py
│   ├── lr_scheduler_integration.py
│   └── full_update_drop.py
│
├── tests/
│   ├── __init__.py
│   ├── test_dropgrad.py
│   └── test_dropgrad_scheduler.py
│
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Installation

The PyTorch implementation of DropGrad can be installed simply using pip or by cloning the current GitHub repo.

### Requirements

The only requirement for DropGrad is PyTorch. (Only versions of PyTorch >= 1.9.0 have been tested, although DropGrad should be compatible with any version of PyTorch)

### Using pip

To install using pip:

```bash
pip install dropgrad
```

### Using git

```bash
git clone https://github.com/dingo-actual/dropgrad.git
cd dropgrad
pip install -r requirements.txt
pip install .
```

## Usage

### Basic Usage

To use DropGrad in your neural network optimization, simply import the `DropGrad` class and wrap your optimizer:

```python
from dropgrad import DropGrad

opt_unwrapped = Adam(net.parameters(), lr=1e-3)
opt = DropGrad(opt_unwrapped, drop_rate=0.1)
```

During training, call `.step()` on the wrapped optimizer to apply DropGrad, and then call `.zero_grad()` to reset the gradients:

```python
opt.step()
opt.zero_grad()
```

### Drop Rate Schedulers

DropGrad supports drop rate schedulers to dynamically adjust the drop rate during training. The package provides several built-in schedulers, including `LinearDropRateScheduler`, `CosineAnnealingDropRateScheduler`, and `StepDropRateScheduler`. To use a drop rate scheduler, pass an instance of a scheduler to the `DropGrad` constructor:

```python
from dropgrad import DropGrad, LinearDropRateScheduler

scheduler = LinearDropRateScheduler(initial_drop_rate=0.1, final_drop_rate=0.0, num_steps=1000)
opt = DropGrad(opt_unwrapped, drop_rate_scheduler=scheduler)
```

### Full Update Drop

DropGrad provides an option to apply "full" update drop by interrupting the `.step()` method. To enable this feature, pass `full_update_drop=True` to the `DropGrad` constructor:

```python
opt = DropGrad(opt_unwrapped, drop_rate=0.1, full_update_drop=True)
```

### Varying Drop Rates per Parameter

DropGrad allows specifying different drop rates for individual parameters or parameter groups. This enables fine-grained control over the regularization applied to different parts of the model. To vary drop rates per parameter, pass a dictionary mapping parameters to drop rates:

```python
params = {
    'encoder': 0.1,
    'decoder': 0.2
}
opt = DropGrad(opt_unwrapped, params=params)
```

## Examples

The `examples` directory contains sample code demonstrating various use cases of DropGrad, including basic usage, integration with learning rate schedulers, and applying full update drop.

## Testing

DropGrad includes a test suite to ensure the correctness of the implementation. The tests cover the functionality of the `DropGrad` optimizer and the drop rate schedulers. To run the tests, use the following command:

```bash
pytest tests/
```

## Analysis

For a detailed analysis of the DropGrad method, including its theoretical foundations, advantages, and empirical results, please refer to the `docs/analysis.md` file.

## Contributing

Contributions to DropGrad are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License

DropGrad is released under the MIT License. See the `LICENSE` file for more details.