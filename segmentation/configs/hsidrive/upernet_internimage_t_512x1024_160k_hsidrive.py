# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/hsi_drive.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_jon.py'
]

data_preprocessor = dict(type='SegDataPreProcessor', mean=None, std=None, bgr_to_rgb=None, pad_val=0, seg_pad_val=255)

#pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_t_1k_224.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        in_channels = 25,
        channels=64,
        depths=[4, 4, 18, 4],
        groups=[4, 8, 16, 32],
        mlp_ratio=4.,
        drop_path_rate=0.2,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=False,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        #init_cfg=dict(type='Pretrained', checkpoint=pretrained)
        ),
    decode_head=dict(num_classes=6, in_channels=[64, 128, 256, 512]),
    auxiliary_head=dict(num_classes=6, in_channels=256),
    test_cfg=dict(mode='whole')
)
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=30, layer_decay_rate=1.0,
                       depths=[4, 4, 18, 4]))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=8)
# fp16 = dict(loss_scale=dict(init_scale=512))
