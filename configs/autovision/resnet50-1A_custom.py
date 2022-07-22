model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='TIMMBackbone',
        model_name='resnet50',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://s3plus.sankuai.com/v1/mss_9240d97c6bf34ab1b78859c3c2a2a3e4/automl-model-zoo/models/resnet/resnet50_a1_0-14fe96d1.pth'
        )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=10,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_soft=True)),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ],
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=10, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=10, prob=0.5)
    ]))
find_unused_parameters = True
policy_imagenet = [[{
    'type': 'Posterize',
    'bits': 4,
    'prob': 0.4
}, {
    'type': 'Rotate',
    'angle': 30.0,
    'prob': 0.6
}],
    [{
        'type': 'Solarize',
        'thr': 113.77777777777777,
        'prob': 0.6
    }, {
        'type': 'AutoContrast',
        'prob': 0.6
    }],
    [{
        'type': 'Equalize',
        'prob': 0.8
    }, {
        'type': 'Equalize',
        'prob': 0.6
    }],
    [{
        'type': 'Posterize',
        'bits': 5,
        'prob': 0.6
    }, {
        'type': 'Posterize',
        'bits': 5,
        'prob': 0.6
    }],
    [{
        'type': 'Equalize',
        'prob': 0.4
    }, {
        'type': 'Solarize',
        'thr': 142.22222222222223,
        'prob': 0.2
    }],
    [{
        'type': 'Equalize',
        'prob': 0.4
    }, {
        'type': 'Rotate',
        'angle': 26.666666666666668,
        'prob': 0.8
    }],
    [{
        'type': 'Solarize',
        'thr': 170.66666666666666,
        'prob': 0.6
    }, {
        'type': 'Equalize',
        'prob': 0.6
    }],
    [{
        'type': 'Posterize',
        'bits': 6,
        'prob': 0.8
    }, {
        'type': 'Equalize',
        'prob': 1.0
    }],
    [{
        'type': 'Rotate',
        'angle': 10.0,
        'prob': 0.2
    }, {
        'type': 'Solarize',
        'thr': 28.444444444444443,
        'prob': 0.6
    }],
    [{
        'type': 'Equalize',
        'prob': 0.6
    }, {
        'type': 'Posterize',
        'bits': 5,
        'prob': 0.4
    }],
    [{
        'type': 'Rotate',
        'angle': 26.666666666666668,
        'prob': 0.8
    }, {
        'type': 'ColorTransform',
        'magnitude': 0.0,
        'prob': 0.4
    }],
    [{
        'type': 'Rotate',
        'angle': 30.0,
        'prob': 0.4
    }, {
        'type': 'Equalize',
        'prob': 0.6
    }],
    [{
        'type': 'Equalize',
        'prob': 0.0
    }, {
        'type': 'Equalize',
        'prob': 0.8
    }],
    [{
        'type': 'Invert',
        'prob': 0.6
    }, {
        'type': 'Equalize',
        'prob': 1.0
    }],
    [{
        'type': 'ColorTransform',
        'magnitude': 0.4,
        'prob': 0.6
    }, {
        'type': 'Contrast',
        'magnitude': 0.8,
        'prob': 1.0
    }],
    [{
        'type': 'Rotate',
        'angle': 26.666666666666668,
        'prob': 0.8
    }, {
        'type': 'ColorTransform',
        'magnitude': 0.2,
        'prob': 1.0
    }],
    [{
        'type': 'ColorTransform',
        'magnitude': 0.8,
        'prob': 0.8
    }, {
        'type': 'Solarize',
        'thr': 56.888888888888886,
        'prob': 0.8
    }],
    [{
        'type': 'Sharpness',
        'magnitude': 0.7,
        'prob': 0.4
    }, {
        'type': 'Invert',
        'prob': 0.6
    }],
    [{
        'type': 'Shear',
        'magnitude': 0.16666666666666666,
        'prob': 0.6,
        'direction': 'horizontal'
    }, {
        'type': 'Equalize',
        'prob': 1.0
    }],
    [{
        'type': 'ColorTransform',
        'magnitude': 0.0,
        'prob': 0.4
    }, {
        'type': 'Equalize',
        'prob': 0.6
    }],
    [{
        'type': 'Equalize',
        'prob': 0.4
    }, {
        'type': 'Solarize',
        'thr': 142.22222222222223,
        'prob': 0.2
    }],
    [{
        'type': 'Solarize',
        'thr': 113.77777777777777,
        'prob': 0.6
    }, {
        'type': 'AutoContrast',
        'prob': 0.6
    }],
    [{
        'type': 'Invert',
        'prob': 0.6
    }, {
        'type': 'Equalize',
        'prob': 1.0
    }],
    [{
        'type': 'ColorTransform',
        'magnitude': 0.4,
        'prob': 0.6
    }, {
        'type': 'Contrast',
        'magnitude': 0.8,
        'prob': 1.0
    }],
    [{
        'type': 'Equalize',
        'prob': 0.8
    }, {
        'type': 'Equalize',
        'prob': 0.6
    }]]
rand_increasing_policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(type='Invert'),
    dict(type='Rotate', magnitude_key='angle', magnitude_range=(0, 30)),
    dict(type='Posterize', magnitude_key='bits', magnitude_range=(4, 0)),
    dict(type='Solarize', magnitude_key='thr', magnitude_range=(256, 0)),
    dict(
        type='SolarizeAdd',
        magnitude_key='magnitude',
        magnitude_range=(0, 110)),
    dict(
        type='ColorTransform',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(type='Contrast', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Brightness', magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(
        type='Sharpness', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='horizontal'),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='vertical'),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        direction='horizontal'),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        direction='vertical')
]
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies=rand_increasing_policies,
        num_policies=2,
        total_level=10,
        magnitude_level=5,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bicubic')
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_prefix='/',
        ann_file=None,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix='/',
        ann_file=None,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix='/',
        ann_file=None,
        pipeline=test_pipeline
    )
)

evaluation = dict(interval=1, metric='accuracy')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.1,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
