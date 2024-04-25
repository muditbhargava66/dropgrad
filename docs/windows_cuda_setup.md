# Setting up CUDA on Windows for PyTorch and DropGrad

This guide will walk you through the steps to set up CUDA on Windows for working with PyTorch and DropGrad.

## Prerequisites

- Windows operating system
- NVIDIA GPU with CUDA support

## Step 1: Install CUDA Toolkit

1. Visit the [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11) page.
2. Select your specific Windows version and architecture.
3. Download the CUDA Toolkit installer.
4. Run the installer and follow the installation instructions.

## Step 2: Install PyTorch with CUDA Support

To install PyTorch with CUDA support, run the following command:

```bash
pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118
```

This command will install the latest version of PyTorch with CUDA 11.8 support.

**Note:** If you are building DropGrad locally, ensure that the `requirements.txt` file includes the appropriate PyTorch version with CUDA support. Update the file if necessary.

## Step 3: Create a Virtual Environment

It is recommended to work within a virtual environment to manage dependencies and avoid conflicts. To create and activate a virtual environment, follow these steps:

```bash
python -m venv new_env
.\new_env\Scripts\activate
```

To deactivate the virtual environment when you're done, simply run:

```bash
deactivate
```

## Step 4: Verify CUDA and NVIDIA Driver Installation

To verify that CUDA and the NVIDIA driver are properly installed, run the following commands:

```bash
nvidia-smi
nvcc --version
```

The `nvidia-smi` command displays information about your NVIDIA GPU and the installed driver version. The `nvcc --version` command shows the installed CUDA version.

If you have multiple GPUs and want to specify which one to use, set the `CUDA_VISIBLE_DEVICES` environment variable:

```bash
set CUDA_VISIBLE_DEVICES=0
```

Replace `0` with the desired GPU index.

## Step 5: Verify PyTorch Installation with CUDA Support

To verify that PyTorch is installed correctly with CUDA support, run the following command:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If the output shows the PyTorch version and `True` for CUDA availability, then PyTorch with CUDA support is properly installed.

## Conclusion

You have now successfully set up CUDA on Windows for working with PyTorch and DropGrad. Make sure to activate your virtual environment and ensure that the `requirements.txt` file includes the appropriate PyTorch version with CUDA support when building DropGrad locally.

If you encounter any issues or have further questions, please refer to the official PyTorch and NVIDIA CUDA documentation or seek support from the community.