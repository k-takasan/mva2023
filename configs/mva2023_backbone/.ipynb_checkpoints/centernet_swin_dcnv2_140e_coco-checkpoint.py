_base_ = [
    # '../_base_/datasets/coco_detection.py',
    './drone_dataset.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    type='CenterNet',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    
    neck=dict(
        type='CTResNetNeck',
        in_channel=768, #
        num_deconv_filters=(256, 128, 64),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=True),

    # neck=dict(
    #     type='FPN',
    #     in_channels=[96, 192, 384, 768],
    #     out_channels=256,
    #     num_outs=5
    # ),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=1,  
        in_channel=64, #
        feat_channel=64, #
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
evaluation = dict(interval=1, metric='bbox') #
checkpoint_config = dict(interval=1) #
