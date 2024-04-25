# DropGrad: A Simple Method for Regularization and Accelerated Optimization of Neural Networks

DropGrad is a regularization method for neural networks that works by randomly (and independently) setting gradient values to zero before an optimization step. Similarly to Dropout, it has a single parameter, `drop_rate`, the probability of setting each parameter gradient to zero. In order to de-bias the remaining gradient values, they are divided by `1.0 - drop_rate`.

## Features

- Simple and easy-to-use gradient regularization technique
- Compatible with various optimizers and learning rate schedulers
- Supports per-parameter drop rates for fine-grained control
- Implements drop rate schedulers for dynamic regularization
- Provides an option to apply "full" update drop for further regularization
- Utilizes mixed-precision training for improved performance and memory efficiency (CUDA devices only)
- Cross-platform compatibility: Works seamlessly on macOS, Windows, and Linux

## Updates in Version 3.0.0

- Enhanced cross-platform compatibility: The codebase now works seamlessly on macOS, Windows, and Linux
- Improved device selection logic: Automatically detects and utilizes the available hardware (MPS, CUDA, or CPU) for training
- Updated dependencies: Added `torchvision`, `torchaudio`, `matplotlib`, and `scipy` as dependencies in `requirements.txt` and `pyproject.toml`
- Improved visualization: Enhanced `visualize.py` with better plot layout and cross-platform file paths
- Code cleanup and refactoring: Improved code structure and readability
- Added mathematical analysis: Introduced `mathematical_analysis.py` to analyze the effect of DropGrad on various optimizers
- Added benchmark visualizations: Introduced `benchmark_visualizations.py` to compare the behavior of DropGrad across optimizers and benchmarks

## Code Structure

```
dropgrad/
│
├── docs/
│   ├── analysis.md
│   └── windows_cuda_setup.md
│
├── dropgrad/
│   ├── __init__.py
│   ├── dropgrad_opt.py
│   └── dropgrad_scheduler.py
│
├── examples/
│   ├── basic_usage.py
│   ├── lr_scheduler_integration.py
│   ├── full_update_drop.py
│   └── vit_experiments/
│       ├── vit_model.py
│       ├── train.py
│       ├── visualize.py
│       ├── mathematical_analysis.py
│       └── benchmark_visualizations.py
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

### Requirements

- Python >= 3.7
- PyTorch >= 1.12.0
- torchvision >= 0.13.0
- torchaudio >= 0.12.0
- matplotlib
- scipy

### Using pip

To install DropGrad using pip, run the following command:

```bash
pip install dropgrad
```

### From source

To install DropGrad from source, follow these steps:

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

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = DropGrad(optimizer, drop_rate=0.1)
```

During training, call `.step()` on the wrapped optimizer to apply DropGrad, and then call `.zero_grad()` to reset the gradients:

```python
optimizer.step()
optimizer.zero_grad()
```

### Drop Rate Schedulers

DropGrad supports drop rate schedulers to dynamically adjust the drop rate during training. The package provides several built-in schedulers, including `LinearDropRateScheduler`, `CosineAnnealingDropRateScheduler`, and `StepDropRateScheduler`. To use a drop rate scheduler, pass an instance of a scheduler to the `DropGrad` constructor:

```python
from dropgrad import DropGrad, LinearDropRateScheduler

scheduler = LinearDropRateScheduler(initial_drop_rate=0.1, final_drop_rate=0.0, num_steps=1000)
optimizer = DropGrad(optimizer, drop_rate_scheduler=scheduler)
```

### Full Update Drop

DropGrad provides an option to apply "full" update drop by interrupting the `.step()` method. To enable this feature, pass `full_update_drop=True` to the `DropGrad` constructor:

```python
optimizer = DropGrad(optimizer, drop_rate=0.1, full_update_drop=True)
```

### Varying Drop Rates per Parameter

DropGrad allows specifying different drop rates for individual parameters or parameter groups. This enables fine-grained control over the regularization applied to different parts of the model. To vary drop rates per parameter, pass a dictionary mapping parameters to drop rates:

```python
params = {
    'encoder': 0.1,
    'decoder': 0.2
}
optimizer = DropGrad(optimizer, params=params)
```

## Examples

The `examples` directory contains sample code demonstrating various use cases of DropGrad, including basic usage, integration with learning rate schedulers, applying full update drop, and training a Vision Transformer (ViT) on the CIFAR-10 dataset under different regularization scenarios.

## Testing

DropGrad includes a test suite to ensure the correctness of the implementation. The tests cover the functionality of the `DropGrad` optimizer and the drop rate schedulers. To run the tests, use the following command:

```bash
pytest tests/
```

## Analysis

For a detailed analysis of the DropGrad method, including its theoretical foundations, advantages, and empirical results, please refer to the `docs/analysis.md` file.

## Windows CUDA Setup

For instructions on setting up CUDA on Windows for PyTorch and DropGrad, please refer to the `docs/windows_cuda_setup.md` file.

## Contributing

Contributions to DropGrad are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License

DropGrad is released under the MIT License. See the [MIT License](LICENSE) file for more details.

## Star History

<a href="https://star-history.com/#muditbhargava66/dropgrad&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=muditbhargava66/dropgrad&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=muditbhargava66/dropgrad&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=muditbhargava66/dropgrad&type=Date" />
 </picture>
</a>