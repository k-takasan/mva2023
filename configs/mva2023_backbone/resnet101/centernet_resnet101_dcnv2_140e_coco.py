_base_ = [
    # '../_base_/datasets/coco_detection.py',
    './drone_dataset.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

model = dict(
    type='CenterNet',
    backbone=dict(
        #type='HourglassNet',
        type='ResNet',
        depth=101, # 18
        norm_eval=False,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101') # 18
        
        #downsample_times=5,
        #num_stacks=2,
        #stage_channels=(256, 256, 384, 384, 384, 512),
        #stage_blocks=(2, 2, 2, 2, 2, 4),
        #feat_channel=512, # 256
        #norm_cfg=dict(type='BN', requires_grad=True),
        #pretrained=None,
        #init_cfg=None
        ),
    neck=dict(
        type='CTResNetNeck',
        in_channel=2048, # 512
        num_deconv_filters=(256, 128, 64),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=True),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=1,  # 80
        in_channel=64,
        feat_channel=64,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100))

# optimizer
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
# Based on the default settings of modern detectors, we added warmup settings.
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[90, 120])
runner = dict(max_epochs=140)

# Avoid evaluation and saving weights too frequently
evaluation = dict(interval=5, metric='bbox')
checkpoint_config = dict(interval=5)
