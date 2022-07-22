_base_ = [
    '../../_base_/datasets/imagenet_bs32.py',
    '../../_base_/schedules/imagenet_bs256.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.5),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1920,
        loss=dict(
            type='LabelSmoothLoss',
            mode='original',
            label_smooth_val=0.1,
            loss_weight=1.0),
        topk=(1, 5),
    ))

algorithm = dict(
    task='Pruning',
    type='AutoSlim',
    architecture=dict(type='MMClsArchitecture', model=model),
    distiller=dict(
        type='SelfDistiller',
        components=[
            dict(
                student_module='head.fc',
                teacher_module='head.fc',
                losses=[
                    dict(
                        type='KLDivergence',
                        name='loss_kd',
                        tau=1,
                        loss_weight=1,
                    )
                ]),
        ]),
    pruner=dict(
        type='RatioPruner',
        ratios=(2 / 12, 3 / 12, 4 / 12, 5 / 12, 6 / 12, 7 / 12, 8 / 12, 9 / 12,
                10 / 12, 11 / 12, 1.0)),
    retraining=False,
    bn_training_mode=True,
    input_shape=None)

data = dict(
    train=dict(
        data_prefix='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/imagenet/train',
        # pipeline=train_pipeline
        ),
    val=dict(
        data_prefix='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/imagenet/val',
        ann_file='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/imagenet/val.txt',
        # pipeline=test_pipeline
    ),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        data_prefix='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/imagenet/val',
        ann_file='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/imagenet/val.txt',
        # pipeline=test_pipeline
    ),
    )

runner = dict(type='EpochBasedRunner', max_epochs=50)
use_ddp_wrapper = True
