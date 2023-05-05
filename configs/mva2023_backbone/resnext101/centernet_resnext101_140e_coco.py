_base_ = './centernet_resnext101_dcnv2_140e_coco.py'

model = dict(neck=dict(use_dcn=False))
