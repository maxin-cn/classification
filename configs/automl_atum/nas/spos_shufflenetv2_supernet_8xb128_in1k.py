_base_ = [
    '../../_base_/datasets/imagenet_bs32.py',
    '../../_base_/schedules/imagenet_bs256.py',
    '../../_base_/default_runtime.py'
]

norm_cfg = dict(type='BN')
model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='SearchableShuffleNetV2', widen_factor=1.0, norm_cfg=norm_cfg),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5),
    ),
)

mutator = dict(
    type='OneShotMutator',
    placeholder_mapping=dict(
        all_blocks=dict(
            type='OneShotOP',
            choices=dict(
                shuffle_3x3=dict(
                    type='ShuffleBlock', kernel_size=3, norm_cfg=norm_cfg),
                shuffle_5x5=dict(
                    type='ShuffleBlock', kernel_size=5, norm_cfg=norm_cfg),
                shuffle_7x7=dict(
                    type='ShuffleBlock', kernel_size=7, norm_cfg=norm_cfg),
                shuffle_xception=dict(
                    type='ShuffleXception', norm_cfg=norm_cfg),
            ))))

algorithm = dict(
    task='NAS',
    type='SPOS',
    architecture=dict(
        type='MMClsArchitecture',
        model=model,
    ),
    mutator=mutator,
    distiller=None,
    retraining=False,
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    train=dict(
        data_prefix='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/imagenet/train',
        pipeline=train_pipeline),
    val=dict(
        data_prefix='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/imagenet/val',
        ann_file='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/imagenet/val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        data_prefix='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/imagenet/val',
        ann_file='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/imagenet/val.txt',
        pipeline=test_pipeline),
    )

runner = dict(max_iters=150000)
evaluation = dict(interval=1000, metric='accuracy')

# checkpoint saving
checkpoint_config = dict(interval=1000)

find_unused_parameters = True
