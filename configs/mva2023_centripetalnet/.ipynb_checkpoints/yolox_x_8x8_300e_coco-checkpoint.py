_base_ = './yolox_s_8x8_300e_coco.py'

load_from = 'work_dirs/yolox_140e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'
# model settings
model = dict(
    backbone=dict(deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
    bbox_head=dict(in_channels=320, feat_channels=320))
