<img src="https://s4.ax1x.com/2022/01/03/THE0u6.png" alt="nebulae-icon.png" width = "215" height = "219" />

# Nebulae Brochure

**A deep learning framework based on PyTorch and concurrent image processing libraries. It aims to offer a set of useful tools and functions.**

------

## Installation

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

## Spotlight

**Dash Board**

It shows fancy progress bars and draw metric curves in real-time.

[![THAIh9.gif](https://s4.ax1x.com/2022/01/03/THAIh9.gif)](https://imgtu.com/i/THAIh9)



**High Compatibility**

Users are able to build networks using Nebulae with PyTorch seamlessly.
