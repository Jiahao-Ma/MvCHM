<p align="center">

  <h1 align="center"> Multiview Detection with Cardboard Human Modeling </h1>
  
  <h3 align="center"> ACCV2024 </h3>

</p>

<h3 align="center"><a href="https://arxiv.org/pdf/2207.02013">Paper</a> | <a href="">Video (Coming soon)</a></h3>

<p align="center">
  <a href="https://github.com/Jiahao-Ma/MvCHM/blob/main/demo/Wildtrack_pc_result.avi">
    <img src="https://raw.githubusercontent.com/Jiahao-Ma/MvCHM/main/demo/Wildtrack_pc_result.gif" alt="teaser" width="75%">
  </a>
</p>

<p align="left">
The project provides official implementation of <a href="https://arxiv.org/pdf/2207.02013">MvCHM</a> in ACCV'24. The paper introduces a multiview pedestrian detection method using "cardboard human modeling," which aggregates 3D point clouds from multiple camera views. This approach improves accuracy by considering human appearance and height, reducing projection errors compared to traditional 2D methods. 
</p>

## TODOs
- [x] Inference and Training Codes
- [ ] Pretrained Models
- [ ] Supplementary dataset Wildtrack+ and MultiviewX+

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
gir clone git@github.com:Jiahao-Ma/MvCHM.git
```