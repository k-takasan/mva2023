#!/usr/bin/env bash
export TORCH_DISTRIBUTED_DEBUG=INFO 
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
GPU_NUM=4


###############################
# Save the prediction results from each model.
###############################
echo "###############################"
echo "Save the prediction results from each model."
echo "###############################"

# CenterNet ResNeXt101
# bash tools/dist_test.sh configs/mva2023_backbone/resnext101/centernet_resnext101_140e_coco_inference.py work_dirs/centernet_resnext101_140e_coco_finetune_large/epoch_2.pth $GPU_NUM --format-only --eval-options jsonfile_prefix=results
# mkdir -p submit/centernet_resnext101
# mv submit/results.bbox.json submit/centernet_resnext101

python SAHI.py configs/mva2023_backbone/resnext101/centernet_resnext101_140e_coco_inference.py work_dirs/centernet_resnext101_140e_coco_finetune_large/epoch_2.pth 512 512 0 centernet_resnext101 &

# CenterNet ResNet101
# bash tools/dist_test.sh configs/mva2023_backbone/resnet101/centernet_resnet101_140e_coco_inference.py work_dirs/centernet_resnet101_140e_coco_hard_negative_training/epoch_20.pth $GPU_NUM --format-only --eval-options jsonfile_prefix=results
# mkdir -p submit/centernet_resnet101
# mv submit/results.bbox.json submit/centernet_resnet101

python SAHI.py configs/mva2023_backbone/resnet101/centernet_resnet101_140e_coco_inference.py work_dirs/centernet_resnet101_140e_coco_hard_negative_training/epoch_20.pth 512 512 1 centernet_resnet101 & 

# d-DETR ResNet50
# bash tools/dist_test.sh configs/mva2023_detr/detr_140e_coco_inference.py work_dirs/detr_140e_coco_finetune_wo_negative/epoch_20.pth $GPU_NUM --format-only --eval-options jsonfile_prefix=results
# mkdir -p submit/detr_resnet50
# mv submit/results.bbox.json submit/detr_resnet50

python SAHI.py configs/mva2023_detr/detr_140e_coco_inference.py work_dirs/detr_140e_coco_finetune_wo_negative/epoch_20.pth 1620 2880 2 detr_resnet50 &

# d-DETR ResNet101
# bash tools/dist_test.sh configs/mva2023_detr_resnext101/detr_resnext101_140e_coco_inference.py work_dirs/detr_resnet101_hard_negative_training/epoch_20.pth $GPU_NUM --format-only --eval-options jsonfile_prefix=results
# mkdir -p submit/detr_resnet50
# mv submit/results.bbox.json submit/detr_resnet101

python SAHI.py configs/mva2023_detr_resnext101/detr_resnext101_140e_coco_inference.py work_dirs/detr_resnet101_hard_negative_training/epoch_20.pth 1620 2880 3 detr_resnet101 

# CentripetalNet Hourglass104
bash tools/dist_test.sh configs/mva2023_centripetalnet/centripetalnet_140e_coco_inference.py work_dirs/centripetalnet_140e_coco_finetune/epoch_40.pth $GPU_NUM --format-only --eval-options jsonfile_prefix=results

mkdir -p submit/centripetalnet_hourglass104
mv submit/results.bbox.json submit/centripetalnet_hourglass104

###############################
# Ensemble the prediction results with WBF.
###############################
echo "###############################"
echo "Ensemble the prediction results with WBF."
echo "###############################"
cd submit/
python WBF.py
