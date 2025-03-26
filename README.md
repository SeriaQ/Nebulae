<p align="center">
  <img src="https://s4.ax1x.com/2022/01/03/THE0u6.png" alt="nebulae-icon.png" width = "128" height = "128" />
</p>


<h1 align="center">Nebulae</h1>

<p align="center">
  <b> A deep learning framework based on PyTorch and prevalent image processing libraries </b><br>
  It aims to offer a set of useful tools and functions.
</p>


<p align="center">
  <img src="https://img.shields.io/pypi/v/nebulae?color=blue&label=Version" alt="Version">
  <img src="https://img.shields.io/github/stars/SeriaQ/Nebulae?style=social" alt="GitHub Repo Stars">
  <img src="https://img.shields.io/badge/Made%20with-Python-blue" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
</p>


------

## 🚀 Spotlight

Several frequently used features are integrated with simple interfaces.

🛠️ **Integrated Feats** - EMA module, multi-source datasets, training log plotting and timer etc.

🎯 **Simplified API** - Unified API for distributed and single-GPU training.

⚡️ **Efficiency** - Data augmentations are reimplemented using Numpy which is faster than PIL.

🧩 **High Compatibility** - Users are able to build networks using Nebulae with PyTorch seamlessly.

------

## ⚡ Quick Start

📸 **Utility**

Obtain GPU stats after some training epochs.

```python
import nebulae as neb
from nebulae import *

gu = kit.GPUtil()
gu.monitor()
for epoch in range(10):
  # --- training code --- #
gu.status()
```

[![OSFsSD.jpg](https://ooo.0x0.ooo/2025/03/16/OSFsSD.jpg)](https://img.tg/image/OSFsSD)



Automatically select unoccupied GPUs. It is useful for a shared machine. 

```python
import nebulae as neb
from nebulae import *
# select 4 GPUs with 2GB or more memory left
engine = npe(device=power.GPU, ngpu=4, least_mem=2048)
```



Find entire distributed training and test code in ./examples/demo_core.py

------

## 📦 Installation

Users can install nebulae from pip

```sh
pip install nebulae
```

For better development, building from Dockerfile is also available. Modifying the libs version and have nvidia-docker on your machine is recommended.

```sh
sudo docker build -t nebulae:std -f Dockerfile.std .
sudo docker run -it --gpus all --ipc=host --ulimit memlock=-1 nebulae:std
```

The latest version supports PyTorch1.6 and above

------

## ❤️ Support

If you find Nebulae helpful, consider giving it a ⭐ on GitHub! ▶️ https://github.com/SeriaQ/Nebulae