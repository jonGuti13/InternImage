train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg'))
]
test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
        type='MultiScaleFlipAug',
        img_scale=(208, 400),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            #dict(type='Resize', keep_ratio=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='HSIDrive20',
        data_root='/data/HSI-Drive/2.0/openmm/',
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline),
    val=dict(
        type='HSIDrive20',
        data_root='/data/HSI-Drive/2.0/openmm/',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type='HSIDrive20',
        data_root='/data/HSI-Drive/2.0/openmm/',
        img_dir='images/test',
        ann_dir='annotations/test',
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], ignore_index=0)
#test_evaluator = val_evaluator