_base_ = [
    '../../_base_/datasets/cifar10_bs16.py',
    '../../_base_/schedules/cifar10_bs128.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

data = dict(
    samples_per_gpu=256,
    workers_per_gpu=8,
    train=dict(
        data_prefix='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/datasets/',
        ),
    val=dict(
        data_prefix='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/datasets/',),
    test=dict(
        data_prefix='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/datasets/'))
algorithm = dict(
    type='IterativePruning', task='Pruning', retraining=False, input_shape=(3, 224, 224),
    architecture=dict(type='MMClsArchitecture', model=model),
    pruner=dict(type='L2Pruner', pruning_ratio=0.5, num_iterations=10))
custom_hooks = [
    dict(
        type='IterativePruningHook',
        interval=1,
        load_from='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/chek/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth',
        # load_from=None,
        deploy_from=None
    ),
]
# lr_config = dict(policy='step', step=[50, 70])
lr_config = dict(
    policy='PruningTwoStage',
    step=[15, 25, 70, 85],
    start_finetune_epoch=30,
    warmup='exp',
    warmup_iters=5,
    warmup_ratio=0.01,
    warmup_by_epoch=True
)
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config=dict(_delete_=True,)
evaluation = dict(interval=1, metric='accuracy', save_best='auto',start=30)
use_ddp_wrapper = True