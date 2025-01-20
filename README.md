# Accelerated PyTorch training on Mac
## Metal acceleration
PyTorch uses the new Metal Performance Shaders (MPS) backend for GPU training acceleration. This MPS backend extends the PyTorch framework, providing scripts and capabilities to set up and run operations on Mac. The MPS framework optimizes compute performance with kernels that are fine-tuned for the unique characteristics of each Metal GPU family. The new mps device maps machine learning computational graphs and primitives on the MPS Graph framework and tuned kernels provided by MPS.
## Requirements
Mac computers with Apple silicon or AMD GPUs
macOS 12.3 or later
Python 3.7 or later
Xcode command-line tools: xcode-select --install
## Get started
You can use either Anaconda or pip. Please note that environment setup will differ between a Mac with Apple silicon and a Mac with Intel x86.
Use the PyTorch installation selector on the installation page to choose Preview (Nightly) for MPS device acceleration. The MPS backend support is part of the PyTorch 1.12 official release. The Preview (Nightly) build of PyTorch will provide the latest mps support on your device.
### 1. Set up
#### Anaconda
Apple silicon
```shell
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
sh Miniconda3-latest-MacOSX-arm64.sh
```
x86
```shell
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
sh Miniconda3-latest-MacOSX-x86_64.sh
```
Creating an environment to test if it support Apple silicon GPU.
```shell
conda create -n m3_torch_mnist python=3.9
conda activate m3_torch_mnist
```
#### pip
You can use preinstalled pip3, which comes with macOS. Alternatively, you can install it from the Python website or the Homebrew package manager.
### 2. Install
#### Anaconda
```Python
conda install pytorch torchvision torchaudio -c pytorch-nightly
```
#### pip
```Python
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```
#### Building from source
Building PyTorch with MPS support requires Xcode 13.3.1 or later. You can download the latest public Xcode release on the Mac App Store or the latest beta release on the Mac App Store or the latest beta release on the Apple Developer website. The USE_MPS environment variable controls building PyTorch and includes MPS support.
To build PyTorch, follow the instructions provided on the PyTorch website.
### 3. Verify
You can verify mps support using a simple Python script:
```Python
import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")
```
The output should show:
```Python
tensor([1.], device='mps:0')
```
### 4. Train
Using command bellow to start mnist training, which can be accelerated by apple silicon m3 supported by PyTorch backends(mps), faster than cpu.
```python
python3 ./train.py --data_url ~/code/mnist/data --train_url ~/code/mnist/out --batch-size 64 --epochs 15
```
### Feedback
The MPS backend is in the beta phase, and we’re actively addressing issues and fixing bugs. To report an issue, use the GitHub issue tracker with the label “module: mps”.
