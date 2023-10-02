norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='InternImage',
        core_op='DCNv3',
        in_channels=25,
        channels=64,
        depths=[4, 4, 18, 4],
        groups=[4, 8, 16, 32],
        mlp_ratio=4.0,
        drop_path_rate=0.2,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=False,
        with_cp=False,
        out_indices=(0, 1, 2, 3)),
    decode_head=dict(
        type='UPerHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=True,
            class_weight=[
                0, 0.023432840583372, 0.487242396681053, 0.059020865556711,
                0.296974520051312, 0.133329377127552
            ])),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=True,
            class_weight=[
                0, 0.023432840583372, 0.487242396681053, 0.059020865556711,
                0.296974520051312, 0.133329377127552
            ])),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
img_norm_cfg = dict(
    mean=[
        0.0555, 0.0595, 0.0665, 0.0625, 0.046, 0.0635, 0.0745, 0.0715, 0.0595,
        0.041, 0.058, 0.0675, 0.059, 0.0505, 0.0415, 0.0505, 0.0525, 0.057,
        0.051, 0.045, 0.059, 0.065, 0.066, 0.0625, 0.044
    ],
    std=[
        0.0555, 0.0595, 0.0665, 0.0625, 0.046, 0.0635, 0.0745, 0.0715, 0.0595,
        0.041, 0.058, 0.0675, 0.059, 0.0505, 0.0415, 0.0505, 0.0525, 0.057,
        0.051, 0.045, 0.059, 0.065, 0.066, 0.0625, 0.044
    ],
    to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='Normalize',
        mean=[
            0.0555, 0.0595, 0.0665, 0.0625, 0.046, 0.0635, 0.0745, 0.0715,
            0.0595, 0.041, 0.058, 0.0675, 0.059, 0.0505, 0.0415, 0.0505,
            0.0525, 0.057, 0.051, 0.045, 0.059, 0.065, 0.066, 0.0625, 0.044
        ],
        std=[
            0.0555, 0.0595, 0.0665, 0.0625, 0.046, 0.0635, 0.0745, 0.0715,
            0.0595, 0.041, 0.058, 0.0675, 0.059, 0.0505, 0.0415, 0.0505,
            0.0525, 0.057, 0.051, 0.045, 0.059, 0.065, 0.066, 0.0625, 0.044
        ],
        to_rgb=False),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'img_norm_cfg'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(208, 400),
        flip=False,
        transforms=[
            dict(
                type='Normalize',
                mean=[
                    0.0555, 0.0595, 0.0665, 0.0625, 0.046, 0.0635, 0.0745,
                    0.0715, 0.0595, 0.041, 0.058, 0.0675, 0.059, 0.0505,
                    0.0415, 0.0505, 0.0525, 0.057, 0.051, 0.045, 0.059, 0.065,
                    0.066, 0.0625, 0.044
                ],
                std=[
                    0.0555, 0.0595, 0.0665, 0.0625, 0.046, 0.0635, 0.0745,
                    0.0715, 0.0595, 0.041, 0.058, 0.0675, 0.059, 0.0505,
                    0.0415, 0.0505, 0.0525, 0.057, 0.051, 0.045, 0.059, 0.065,
                    0.066, 0.0625, 0.044
                ],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type='HSIDrive20',
        data_root='/data/',
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Normalize',
                mean=[
                    0.0555, 0.0595, 0.0665, 0.0625, 0.046, 0.0635, 0.0745,
                    0.0715, 0.0595, 0.041, 0.058, 0.0675, 0.059, 0.0505,
                    0.0415, 0.0505, 0.0525, 0.057, 0.051, 0.045, 0.059, 0.065,
                    0.066, 0.0625, 0.044
                ],
                std=[
                    0.0555, 0.0595, 0.0665, 0.0625, 0.046, 0.0635, 0.0745,
                    0.0715, 0.0595, 0.041, 0.058, 0.0675, 0.059, 0.0505,
                    0.0415, 0.0505, 0.0525, 0.057, 0.051, 0.045, 0.059, 0.065,
                    0.066, 0.0625, 0.044
                ],
                to_rgb=False),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_semantic_seg'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor',
                           'img_norm_cfg'))
        ],
        ignore_index=0),
    val=dict(
        type='HSIDrive20',
        data_root='/data/',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(208, 400),
                flip=False,
                transforms=[
                    dict(
                        type='Normalize',
                        mean=[
                            0.0555, 0.0595, 0.0665, 0.0625, 0.046, 0.0635,
                            0.0745, 0.0715, 0.0595, 0.041, 0.058, 0.0675,
                            0.059, 0.0505, 0.0415, 0.0505, 0.0525, 0.057,
                            0.051, 0.045, 0.059, 0.065, 0.066, 0.0625, 0.044
                        ],
                        std=[
                            0.0555, 0.0595, 0.0665, 0.0625, 0.046, 0.0635,
                            0.0745, 0.0715, 0.0595, 0.041, 0.058, 0.0675,
                            0.059, 0.0505, 0.0415, 0.0505, 0.0525, 0.057,
                            0.051, 0.045, 0.059, 0.065, 0.066, 0.0625, 0.044
                        ],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        ignore_index=0),
    test=dict(
        type='HSIDrive20',
        data_root='/data/',
        img_dir='images/test',
        ann_dir='annotations/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(208, 400),
                flip=False,
                transforms=[
                    dict(
                        type='Normalize',
                        mean=[
                            0.0555, 0.0595, 0.0665, 0.0625, 0.046, 0.0635,
                            0.0745, 0.0715, 0.0595, 0.041, 0.058, 0.0675,
                            0.059, 0.0505, 0.0415, 0.0505, 0.0525, 0.057,
                            0.051, 0.045, 0.059, 0.065, 0.066, 0.0625, 0.044
                        ],
                        std=[
                            0.0555, 0.0595, 0.0665, 0.0625, 0.046, 0.0635,
                            0.0745, 0.0715, 0.0595, 0.041, 0.058, 0.0675,
                            0.059, 0.0505, 0.0415, 0.0505, 0.0525, 0.057,
                            0.051, 0.045, 0.059, 0.065, 0.066, 0.0625, 0.044
                        ],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        ignore_index=0))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], ignore_index=0)
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], ignore_index=0)
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=0.01,
    betas=(0.9, 0.999),
    weight_decay=0.001,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=30, layer_decay_rate=1.0, depths=[4, 4, 18, 4]))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.0001,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=140000)
checkpoint_config = dict(by_epoch=False, interval=1400, max_keep_ckpts=1)
evaluation = dict(
    interval=14000, metric='mIoU', pre_eval=True, save_best='mIoU')
work_dir = './work_dirs/upernet_internimage_t_512x1024_160k_hsidrive'
gpu_ids = [0]
auto_resume = False
