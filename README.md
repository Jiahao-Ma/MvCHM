# Multiview Detection with Cardboard Human Modeling
 The official implementation of Multiview Detection with Cardboard Human Modeling. The paper has been accepted by ACCV 2024.

Code and data for paper:  
**Multiview Detection with Cardboard Human Modeling**  
[Jiahao Ma*](https://github.com/Jiahao-Ma), [Zicheng Duan*](https://github.com/ZichengDuan), [Liang Zheng](https://zheng-lab.cecs.anu.edu.au/), [Chuong Nguyen](https://people.csiro.au/N/C/Chuong-Nguyen)  
[[Paper and Appendices](https://arxiv.org/pdf/2207.02013)]

## Overview
We propose a new state-of-the-art method MvCHM for multiview object detection, and some major upgrades on existing multiview pedestrian detectiom benchmarks [Wildtrack](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) dataset and [MultiviewX](https://github.com/hou-yz/MultiviewX) dataset, respectively named as Wildtrack+ and MultiviewX+.

## Prerequisites
This project is tested to run on environment with:  
 - CUDA > 11
 - torchvision 0.11.3
 - Windowns 10/11, Ubuntu 20.04 
### Installation

1. Create a new conda environment named `mvchm` to run this project:
```
conda create -n mvchm python=3.9.7
conda activate mvchm
```

2. Make sure your system meet the CUDA requirements and install some core packages:
```
pip install easydict torch==1.12.1+cu113 torchvision==0.13.1+cu113 tqdm scipy opencv-python
```

3. Clone this repository
```
cd Your-Project-Folder
gir clone git@github.com:ZichengDuan/MvCHM.git
```