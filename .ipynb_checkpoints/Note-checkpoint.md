Small Object Detection Challenge for Spotting Birds 2023
# Overview of the Competition

## Metric
- mAP@50

## Dataset
- Drone2021
    - train:,test:
- mva2023
    - train:,test(public):

# Installation
```
git clone
$ ln -s "path/to/data" data
conda create -n mva python=3.7
conda activate mva
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -U openmim
mim install mmcv-full==1.6.0
pip install -v -e .
```
## directory tree


# Script for training and submission


# Baseline Model (Official)
- Centernet(Backbone:Resnet18)
    - epoch
    - DA, TTA
    - Hard negative
    
# Best Solution
- CenterNet(Backbone:ResNeXt101)+SAHI
    - Config
        - epoch
        - Hard negative
        - DA, merge
        - SAHI kernel size
- d-DETR(Backbone:ResNet50) 
- WBF(Ensemble)

# What I Tried
## Effective
- CenterNet(Backbone:ResNet101)
- Deformable DETR(Backbone:ResNet50)
- Centripetalnet

## Not Effective
- Mask-RCNN(Backbone:Swin Transformer)
- YOLOX

## Failed to Implement
- Simple Copy Paste

# What I did not try
- Semi-Supervised Object Detection(not allowed in this competition)

# Reference
- MVA2023 https://www.mva-org.jp/mva2023/index.php?id=challenge
- Baseline Code https://github.com/IIM-TTIJ/MVA2023SmallObjectDetection4SpottingBirds

- MMDetection's documentation https://mmdetection.readthedocs.io/en/latest/
- SAHI(github) https://github.com/obss/sahi
- Deformable DETR(MMDetection) https://github.com/open-mmlab/mmdetection/blob/master/configs/deformable_detr/README.md
- CentripetalNet(MMDetection) https://github.com/open-mmlab/mmdetection/tree/master/configs/centripetalnet

- Awesome Tiny Object Detection(github) https://github.com/kuanhungchen/awesome-tiny-object-detection
- SmallObjectDetectionList(github) https://github.com/ispc-lab/SmallObjectDetectionList

- G. Chen et al., "A Survey of the Four Pillars for Small Object Detection: Multiscale Representation, Contextual Information, Super-Resolution, and Region Proposal," in IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 52, no. 2, pp. 936-953, Feb. 2022, doi: 10.1109/TSMC.2020.3005231.
- Yu, X. et al. (2020). The 1st Tiny Object Detection Challenge: Methods and Results. In: Bartoli, A., Fusiello, A. (eds) Computer Vision – ECCV 2020 Workshops. ECCV 2020. Lecture Notes in Computer Science(), vol 12539. Springer, Cham. https://doi.org/10.1007/978-3-030-68238-5_23