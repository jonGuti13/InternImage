# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/hsi_drive.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_jon.py'
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

