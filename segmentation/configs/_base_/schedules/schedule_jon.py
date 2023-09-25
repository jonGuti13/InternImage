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

#En los .log.json que se generan cuando lanzo un entrenamiento se va guardando la información de entrenamiento/validación.
#Voy a utilizar el ejemplo de schedule_160k.py para comentar algunas de las opciones.

## optimizer
#optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
#optimizer_config = dict()
## learning policy
#lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
## runtime settings
#runner = dict(type='IterBasedRunner', max_iters=160000)
#checkpoint_config = dict(by_epoch=False, interval=16000, max_keep_ckpts=1) -->        Cada 16000 iteraciones almacena un checkpoint y borra el anterior
#evaluation = dict(interval=16000, metric='mIoU', pre_eval=True, save_best='mIoU') --> Cada 16000 iteraciones se realiza la validación (no tiene por qué coincidir con la finalización de una epoch) y se actualiza (o no) el checkpoint correspondiente al mejor mIoU.

#Por otro lado está el 'log_config'
#log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)]) --> Actualiza los valores en el log.json correspondiente