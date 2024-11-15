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
- [x] Pretrained Models
- [x] Supplementary dataset Wildtrack+ and MultiviewX+

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

3. Clone this repository:
```
cd Your-Project-Folder
gir clone git@github.com:Jiahao-Ma/MvCHM.git
```
4. Download the pretrained checkpoint.

* Download the standing point detection model checkpoint `mspn_mx.pth` and `mspn_wt.pth` from [here](https://drive.google.com/drive/folders/1biDfkpHrVkZ104VANDDyAMaxQMlQ0SpJ?usp=sharing) and put them in `\model\refine\checkpoint`.
* Download the human detection model checkpoint `rcnn_mxp.pth` and `rcnn_wtp.pth` from [here](https://drive.google.com/drive/folders/1biDfkpHrVkZ104VANDDyAMaxQMlQ0SpJ?usp=sharing) and put them in `\model\detector\checkpoint`.



### Inference
Quick start for the project
1. Download the pre-trained checkpoint `Wildtrack.pth` from [here](https://drive.google.com/drive/folders/1biDfkpHrVkZ104VANDDyAMaxQMlQ0SpJ?usp=sharing) and put them in `\checkpoints`.
2. Inference.
```
python inference.py --dataname Wildtrack --data_root /path/to/Wildtrack
```

### Training
Train on Wildtrack dataset. Specify the path to Wildtrack dataset.
```
python train.py --dataname Wildtrack --data_root /path/to/Wildtrack
```

Train on MultiviewX dataset. Specify the path to MultiviewX dataset.
```
python train.py --dataname MultiviewX --data_root /path/to/MultiviewX
```

### Evaluation
Evaluate on the trained model.
```
# Example: --cfg_file: experiments\2022-10-23_19-53-52_wt\MvDDE.yaml
python evaluate.py --dataname Wildtrack --data_root /path/to/Wildtrack --cfg_file /path/to/cfg_file
```

### Wildtrack+ and MultiviewX+ (Optional)
We provide the supplementary datasets Wildtrack+ and MultiviewX+, which include additional annotations for pedestrians located outside the predefined ground plane. You can download the dataset [here](https://drive.google.com/drive/folders/11CtXneqbsqSFCtILfUuD7orvoeSwzYSw?usp=sharing).