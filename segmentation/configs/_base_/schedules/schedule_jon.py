# optimizer
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=30, layer_decay_rate=1.0, depths=[4, 4, 18, 4]))
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
# runtime settings
runner = dict(type="EpochBasedRunner", max_epochs=50)
checkpoint_config = dict(by_epoch=True, interval=10, max_keep_ckpts=1)
evaluation = dict(interval=1, metric='mIoU', save_best='mIoU')