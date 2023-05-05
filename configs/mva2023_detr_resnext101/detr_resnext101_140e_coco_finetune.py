from set_lib_dir import LIB_ROOT_DIR
import os
_base_ = './detr_resnext101_140e_coco.py'
data_root = LIB_ROOT_DIR + '/data/'


data = dict(
    train=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_train_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ),
    val=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_val_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ),
    test=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_val_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    )
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[20, 30])
runner = dict(max_epochs=40)
evaluation = dict(interval=5, metric='bbox')
load_from = LIB_ROOT_DIR + '/work_dirs/detr_resnext101_140e_coco/epoch_50.pth' #
checkpoint_config = dict(interval=5)
