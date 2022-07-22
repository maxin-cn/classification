_base_ = [
    '../_base_/models/resnet50_timm.py', '../_base_/datasets/imagenet_bs64_pil_resize_autoaug.py',
    '../_base_/schedules/imagenet_bs256_200e_coslr_warmup.py', '../_base_/default_runtime.py',
    '../_base_/datasets/pipelines/rand_aug.py'
]
# Model config
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://s3plus.sankuai.com/v1/mss_9240d97c6bf34ab1b78859c3c2a2a3e4/automl-model-zoo/models/resnet/resnet50_a1_0-14fe96d1.pth',
        )),
    head=dict(num_classes=1000),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
    ])
)

# Dataset config
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
        policies={{_base_.rand_increasing_policies}},
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
        data_prefix='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/imagenet/train',
        # ann_file='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/wengkaiheng/dataset/food/food_dataset/train.txt',
        ann_file=None,
        pipeline=train_pipeline
    ),
    val=dict(
        data_prefix='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/imagenet/val',
        # ann_file='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/wengkaiheng/dataset/food/food_dataset/val.txt',
        ann_file=None,
        pipeline=test_pipeline
    ),
    test=dict(
        data_prefix='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/imagenet/test',
        # ann_file='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/wengkaiheng/dataset/food/food_dataset/val.txt',
        ann_file=None,
        pipeline=test_pipeline
    )
)

# Training schedule config
# lr is set for a batch size of 128
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.1,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=200)
log_config = dict(interval=100)
#compress = False
