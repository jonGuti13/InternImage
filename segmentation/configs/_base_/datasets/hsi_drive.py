#Para normalizar los datos al rango [-1, 1] se hace la normalización 2 * ((valor - minimo)/(maximo - minimo)) - 1
#No sé muy bien cómo se implementa esto aquí, pero sé que existe la forma de normalizar haciendo (valor - mean)/std
#La pregunta es, ¿Puedo transformar la expresión de arriba para que sea de la forma de la de abajo de tal manera que
#mean=f(maximo, minimo) y std=(maximo, minimo)?
#Pues sí, desarrollando la expresión inicial llego a (valor - (maximo+minimo)/2) / (maximo - minimo)/2
#Para el caso en el que minimo = 0 (algo habitual):
#mean = maximo/2
#std = maximo/2


img_norm_cfg = dict(
    mean=[0.0555, 0.0595, 0.0665, 0.0625, 0.0460, 0.0635, 0.0745, 0.0715, 0.0595, 0.0410, 0.0580, 0.0675, 0.0590, 0.0505, 0.0415, 0.0505, 0.0525, 0.0570, 0.0510, 0.0450, 0.0590, 0.0650, 0.0660, 0.0625, 0.0440],
    std=[0.0555, 0.0595, 0.0665, 0.0625, 0.0460, 0.0635, 0.0745, 0.0715, 0.0595, 0.0410, 0.0580, 0.0675, 0.0590, 0.0505, 0.0415, 0.0505, 0.0525, 0.0570, 0.0510, 0.0450, 0.0590, 0.0650, 0.0660, 0.0625, 0.0440],
    to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),   #Esto trabaja con (N, C, H, W) y como mis imágenes son (H, W, C) esto lo cambia automáticamente a (C, H, W)
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
            #Esto trabaja con (N, C, H, W) y como mis imágenes son (H, W, C) esto lo cambia automáticamente a (C, H, W)
            dict(type='Normalize', **img_norm_cfg),
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
        pipeline=train_pipeline,
        ignore_index=0),
    val=dict(
        type='HSIDrive20',
        data_root='/data/HSI-Drive/2.0/openmm/',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline,
        ignore_index=0),
    test=dict(
        type='HSIDrive20',
        data_root='/data/HSI-Drive/2.0/openmm/',
        img_dir='images/test',
        ann_dir='annotations/test',
        pipeline=test_pipeline,
        ignore_index=0))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], ignore_index=0)
test_evaluator = val_evaluator