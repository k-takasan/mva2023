from set_lib_dir import LIB_ROOT_DIR
import os
_base_ = './centernet_resnext101_140e_coco.py'
data_root = LIB_ROOT_DIR + '/data/'


data = dict(
    train=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/merged_train.json', #
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ),
    val=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_val_coco.json', #
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ),
    test=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_val_coco.json', #
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    )
)


lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[15, 18])
runner = dict(max_epochs=20)
evaluation = dict(interval=1, metric='bbox') #
load_from = LIB_ROOT_DIR + '/work_dirs/resnext101/centernet_resnext101_140e_coco_hard_negative_training/epoch_20.pth' # 
checkpoint_config = dict(interval=1)
