#!/usr/bin/env python
# coding: utf-8
import sys

# arrange an instance segmentation model for test
from sahi.utils.mmdet import (
    download_mmdet_cascade_mask_rcnn_model,
    download_mmdet_config,
)

# import required functions, classes
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict

args = sys.argv

config_path = str(args[1]) 
model_path = str(args[2]) 

model_type = "mmdet"
model_path = model_path
model_config_path = config_path
model_device = "cuda:{}".format(int(args[5])) #
model_confidence_threshold = 0.10 #

slice_height = int(args[3]) #centernet:512,detr:2160*0.75=1620
slice_width = int(args[4]) #centernet:512,detr:3840*0.75=2880
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

source_image_dir = "./data/mva2023_sod4bird_pub_test/images/"
dataset_json_path = "./data/mva2023_sod4bird_pub_test/annotations/public_test_coco_empty_ann.json"


predict(
    model_type=model_type,
    model_path=model_path,
    model_config_path=config_path,
    model_device=model_device,
    model_confidence_threshold=model_confidence_threshold,
    source=source_image_dir,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
    dataset_json_path = dataset_json_path,
    novisual = True,
    project = "sahi",
    name = str(args[6]),
)






