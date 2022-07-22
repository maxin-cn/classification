from re import T


_base_ = [
    '../../_base_/datasets/imagenet_bs32.py',
    '../../_base_/schedules/imagenet_bs256.py',
    '../../_base_/default_runtime.py'
]



dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
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
   samples_per_gpu=128,
   workers_per_gpu=4,
   train=dict(
       type=dataset_type,
       data_prefix='/home/hadoop-basecv/cephfs/data/maxin/work/dataflow/bicycle_parking_space_v3',
       ann_file='/home/hadoop-basecv/cephfs/data/maxin/work/dataflow/bicycle_parking_space_v3/train.txt',
       pipeline=train_pipeline),
   val=dict(
       type=dataset_type,
       data_prefix='/home/hadoop-basecv/cephfs/data/maxin/work/dataflow/bicycle_parking_space_v3',
       ann_file='/home/hadoop-basecv/cephfs/data/maxin/work/dataflow/bicycle_parking_space_v3/val.txt',
       pipeline=test_pipeline),
   test=dict(
       # replace `data/val` with `data/test` for standard test
       type=dataset_type,
       data_prefix='/home/hadoop-basecv/cephfs/data/maxin/work/dataflow/bicycle_parking_space_v3',
       ann_file='/home/hadoop-basecv/cephfs/data/maxin/work/dataflow/bicycle_parking_space_v3/test.txt',
       pipeline=test_pipeline))

evaluation = dict(interval=5, metric='accuracy')


# model settings
student = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.0, init_cfg=dict(
        type='Pretrained',
        checkpoint='mb2_imagenet.pth',
        prefix='backbone',
    )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
"""
student = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
"""

# teacher settings
teacher_ckpt = '/workdir/meituan/infra-mt-cvzoo-classification/swin_best.pth'  # noqa: E501

teacher = dict(
    type='mmcls.ImageClassifier',
    init_cfg=dict(type='Pretrained', checkpoint=teacher_ckpt),
    backbone=dict(
        type='SwinTransformer', arch='base', img_size=224, drop_path_rate=0.5),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=1024,
        loss=dict(
            type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
"""
teacher = dict(
    type='mmcls.ImageClassifier',
    init_cfg=dict(type='Pretrained', checkpoint=teacher_ckpt),
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
"""

# algorithm setting
algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMClsArchitecture',
        model=student,
    ),
    with_student_loss=True,
    with_teacher_loss=False,
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        teacher_norm_eval=True,
        components=[
            dict(
                student_module='neck.gap',
                teacher_module='neck.gap',
                losses=[
                    dict(
                        type='DistanceWiseRKD',
                        name='distance_wise_loss',
                        loss_weight=25.0,
                        with_l2_norm=True),
                    dict(
                        type='AngleWiseRKD',
                        name='angle_wise_loss',
                        loss_weight=50.0,
                        with_l2_norm=True),
                ])
        ]),
)

find_unused_parameters = True
compress_op = 'kd'
