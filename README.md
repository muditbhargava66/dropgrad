# DropGrad: A Simple Method for Regularization and Accelerated Optimization of Neural Networks

![Version](https://img.shields.io/badge/version-0.3.5-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

DropGrad is a regularization method for neural networks that works by randomly (and independently) setting gradient values to zero before an optimization step. Similarly to Dropout, it has a single parameter, `drop_rate`, the probability of setting each parameter gradient to zero. In order to de-bias the remaining gradient values, they are divided by `1.0 - drop_rate`.

## Table of Contents

   * [Features](#features)
   * [What's New in Version 0.3.5?](#whats-new-in-version-035)
   * [Directory Structure](#Directory-Structure)
   * [Installation](#installation)
   * [Usage](#usage)
   * [Examples](#examples)
   * [Testing](#testing)
   * [Analysis](#analysis)
   * [Windows CUDA Setup](#windows-cuda-setup)
   * [Contributing](#contributing)
   * [License](#license)
   * [Star History](#star-history)

## Features

- Simple and easy-to-use gradient regularization technique
- Compatible with various optimizers and learning rate schedulers
- Supports per-parameter drop rates for fine-grained control
- Implements drop rate schedulers for dynamic regularization
- Provides an option to apply "full" update drop for further regularization
- Utilizes mixed-precision training for improved performance and memory efficiency (CUDA devices only)
- Cross-platform compatibility: Works seamlessly on macOS, Windows, and Linux

## What's New in Version 0.3.5?

- Added support for the Lion optimizer in the ViT experiments
- Implemented gradient clipping to prevent gradient explosion and improve training stability
- Enhanced data augmentation techniques for better model generalization
- Improved error handling and user interruption handling during training
- Updated test suite to cover various aspects of DropGrad, including initialization, optimization step, drop rate scheduling, and saving of loss values
- Code refactoring and documentation enhancements for better readability and maintainability

## Directory Structure

<table>
<thead>
<tr>
<th> Description </th>
<th> Quick Access </th>
</tr>
</thead>
<tbody>

<tr> <td> <h3> Getting Started </h3>
The <code>examples</code> directory contains sample code demonstrating various use cases of DropGrad, including basic usage, integration with learning rate schedulers, applying full update drop, and training a Vision Transformer (ViT) on the CIFAR-10 dataset under different regularization scenarios.
</td> <td> <pre>
└── examples
    ├── <a href="examples/basic_usage.py">basic_usage.py</a>
    ├── <a href="examples/lr_scheduler_integration.py">lr_scheduler_integration.py</a>
    ├── <a href="examples/full_update_drop.py">full_update_drop.py</a>
    └── <a href="examples/vit_experiments">vit_experiments</a>
        ├── <a href="examples/vit_experiments/vit_model.py">vit_model.py</a>
        ├── <a href="examples/vit_experiments/train.py">train.py</a>
        ├── <a href="examples/vit_experiments/visualize.py">visualize.py</a>
        ├── <a href="examples/vit_experiments/mathematical_analysis.py">mathematical_analysis.py</a>
        ├── <a href="examples/vit_experiments/benchmark_visualizations.py">benchmark_visualizations.py</a>
        └── <a href="examples/vit_experiments">*.pth</a>
</pre> </td> </tr>

<tr> <td> <h3> Documentation </h3>
The <code>docs</code> directory contains detailed documentation and analysis of the DropGrad method, as well as instructions for setting up CUDA on Windows for PyTorch and DropGrad.
</td> <td> <pre>
└── docs
    ├── <a href="docs/analysis.md">analysis.md</a>
    └── <a href="docs/windows_cuda_setup.md">windows_cuda_setup.md</a>
</pre> </td> </tr>

<tr> <td> <h3> Core DropGrad Implementation </h3>
The <code>dropgrad</code> directory contains the core implementation of the DropGrad optimizer and drop rate schedulers.
</td> <td> <pre>
└── dropgrad
    ├── <a href="dropgrad/__init__.py">__init__.py</a>
    ├── <a href="dropgrad/dropgrad_opt.py">dropgrad_opt.py</a>
    └── <a href="dropgrad/dropgrad_scheduler.py">dropgrad_scheduler.py</a>
</pre> </td> </tr>

<tr> <td> <h3> Testing </h3>
The <code>tests</code> directory contains the test suite for DropGrad, ensuring the correctness of the implementation. The tests cover the functionality of the <code>DropGrad</code> optimizer and the drop rate schedulers.
</td> <td> <pre>
└── tests
    ├── <a href="tests/__init__.py">__init__.py</a>
    ├── <a href="tests/test_dropgrad.py">test_dropgrad.py</a>
    ├── <a href="tests/test_dropgrad_optimizer.py">test_dropgrad_optimizer.py</a>
    └── <a href="tests/test_dropgrad_scheduler.py">test_dropgrad_scheduler.py</a>
</pre> </td> </tr>

<tr> <td> <h3> Configuration and Setup </h3>
This section highlights the key files related to project configuration, requirements, and licensing.
</td> <td> <pre>
├── <a href=".gitignore">.gitignore</a>
├── <a href="LICENSE">LICENSE</a>
├── <a href="pyproject.toml">pyproject.toml</a>
├── <a href="README.md">README.md</a>
└── <a href="requirements.txt">requirements.txt</a>
</pre> </td> </tr>

</tbody>
</table>

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

DropGrad is released under the MIT License. See the `LICENSE` file for more details.

## Star History

<a href="https://star-history.com/#muditbhargava66/dropgrad&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=muditbhargava66/dropgrad&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=muditbhargava66/dropgrad&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=muditbhargava66/dropgrad&type=Date" />
 </picture>
</a>
