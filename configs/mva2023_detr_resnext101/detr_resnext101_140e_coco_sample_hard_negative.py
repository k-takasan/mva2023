_base_ = './detr_resnext101_140e_coco.py'
data_root = 'data/'

data = dict(
    test=dict(
        samples_per_gpu=2,
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_train_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ) 
)

