# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/hsi_drive.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

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
    decode_head=dict(num_classes=6, in_channels=[64, 128, 256, 512],
                loss_decode=dict(
                    avg_non_ignore=True,
                    class_weight=[
                        0,
                        0.023432840583372,
                        0.487242396681053,
                        0.059020865556711,
                        0.296974520051312,
                        0.1333293771275520
                    ],
                loss_weight=1.0,
                type='CrossEntropyLoss',
                use_sigmoid=False)),
    auxiliary_head=dict(num_classes=6, in_channels=256,
                loss_decode=dict(
                    avg_non_ignore=True,
                    class_weight=[
                        0,
                        0.023432840583372,
                        0.487242396681053,
                        0.059020865556711,
                        0.296974520051312,
                        0.1333293771275520
                    ],
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False)),
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


runner = dict(type='IterBasedRunner')
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=16000, metric='mIoU', save_best='mIoU')

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=32,
          workers_per_gpu=4,
        train=dict(
        data_root='/data/'),
        val=dict(
        data_root='/data/'),
        test=dict(
        data_root='/data/'))

# fp16 = dict(loss_scale=dict(init_scale=512))

