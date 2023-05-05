_base_ = './maskrcnn_swin_140e_coco.py'
data_root = 'data/'

data = dict(
    test=dict(
        samples_per_gpu=4,
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/merged_train.json', #
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ) 
)

