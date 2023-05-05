_base_ = './centernet_resnet101_dcnv2_140e_coco.py'

model = dict(neck=dict(use_dcn=False))
