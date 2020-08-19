## Installation

Our [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
has step-by-step instructions that install detectron2.
The [Dockerfile](docker)
also installs detectron2 with a few simple commands.

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.4 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization


### Build Detectron2 from Source

1. 根据NVIDIA显卡驱动确定自己可以安装的CUDA版本以及对于的pytorch版本
<img src="introduce_materials/cuda_version.png" width="600" >
(上图来自[What's New in CUDA 11.0 GA](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-whats-new))

gcc & g++ ≥ 5 are required. [ninja](https://ninja-build.org/) is recommended for faster build.
After having them, run:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# Or if you are on macOS
CC=clang CXX=clang++ python -m pip install ......
```

Note that:
1. .
2. .

### Common Installation Issues

Click each issue for its solutions:

<details>
<summary>
Undefined C++ symbols (e.g. `GLIBCXX`) or C++ symbols not found.
</summary>
<br/>
Usually it's because the library is compiled with a newer C++ compiler but run with an old C++ runtime.

This often happens with old anaconda.
Try `conda update libgcc`. Then rebuild detectron2.

The fundamental solution is to run the code with proper C++ runtime.
One way is to use `LD_PRELOAD=/path/to/libstdc++.so`.

</details>

