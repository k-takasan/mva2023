# MVA2023 - Small Object Detection Challenge for Spotting Birds

Author: Kazuki Takasan (k-takasan)

This repository includes my submission code used in the [challenge](http://www.mva-org.jp/mva2023/challenge) .

# Overview of my solution
I performed an ensemble of the following five models using WBF[1].
- CenterNet (Backbone:ResNeXt101) + SAHI[2]
- CenterNet (Backbone:ResNet101) + SAHI[2]
- Deformable-DETR (Backbone:ResNet50) + SAHI[2]
- Deformable-DETR (Backbone:ResNet101) + SAHI[2]
- Centripetalnet (Backbone:Hourglass101)

# Requirements
Please refer to env.log for the details.
- OS: Ubuntu20.04LTS
- Python: 3.7.16
- CUDA : 11.3
- GPU: Tesla V100-PCIE-32GB x 4
- GCC: gcc (Debian 10.2.1-6) 10.2.1 20210110
- PyTorch: 1.10.1
- TorchVision: 0.11.2
- OpenCV: 4.7.0
- MMCV: 1.6.0
- MMCV Compiler: GCC 9.3
- MMCV CUDA Compiler: 11.3
- MMDetection: 2.24.1+c146527

# Installation
<!-- 1. Clone this repository(uploading soon)
```
git clone https://github.com/k-takasan/mva2023_submit_takasan.git
``` -->
1. Create a link file to the data directory
```
ln -s "path/to/data" data
```
2. Create a conda environment and activate it.
```
conda create -n mva python=3.7
conda activate mva
```
3. Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/)
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```
4. Install MMCV
```
pip install -U openmim
mim install mmcv-full==1.6.0
pip install -v -e .
```

# Making inference and submission
Inference and submission can be performed by the following command.
```shell
bash ./test_submist.sh
```
In the script, the following steps are performed.
1. Inference on each model with(or without) SAHI
2. Ensemble with WBF
3. Make submission file (./submit/WBF/WBF*.zip)

# Training models
Training models can be performed by the following command (but this script is just an example)
```shell
bash ./train_test.sh
```

Training methods differ depending on the model, but the following steps are common.
1. Pre-training with the Drone2021 dataset
2. Finetuning with the mva2023 dataset
3. Extra finetuning with the methods like hard negative training  

Please refer to each config file for the details.

# References
[1] [Weighted Boxes Fusion: Ensembling Boxes for Object Detection Models](https://arxiv.org/abs/1910.13302)

[2] [SAHI: Self-Adaptive Hierarchical IoU Loss for Small Object Detection](https://arxiv.org/abs/2108.09418)