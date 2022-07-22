custom_imports = dict(imports=['mmcls.core.optimizer.hspg'], allow_failed_imports=False)
custom_imports = dict(imports=['mmcls.core.utils.hspg_optimizer_hook'], allow_failed_imports=False)

_base_ = [
   '../_base_/models/resnet50_oto_imagenet.py',
   '../_base_/datasets/custom_bs64_autoaug.py',
   '../_base_/default_runtime.py'
]

dataset_type = 'ImageNet'
img_norm_cfg = dict(
   mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
   dict(type='LoadImageFromFile'),
   dict(type='RandomResizedCrop', size=224),
   dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
   dict(type='AutoAugment', policies={{_base_.policy_imagenet}}),
   dict(type='Normalize', **img_norm_cfg),
   dict(type='ImageToTensor', keys=['img']),
   dict(type='ToTensor', keys=['gt_label']),
   dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
   dict(type='LoadImageFromFile'),
   dict(type='Resize', size=(224, -1)),
   dict(type='CenterCrop', crop_size=224),
   dict(type='Normalize', **img_norm_cfg),
   dict(type='ImageToTensor', keys=['img']),
   dict(type='Collect', keys=['img'])
]

model = dict(
   backbone=dict(
       init_cfg=dict(
           type='Pretrained',
           checkpoint='http://s3plus.sankuai.com/v1/mss_8cb9a34d9587426fbf4d3f42b8c31c86/basecv-model/models/pytorch/zhangjinjin/models/scl/resnet50_scl_hard_sample_mining.pth.tar'),
       num_classes=2),
   head = dict(
        # type='MultiLabelLinearClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_soft=True)
    ),
   train_cfg=dict(
       augments=dict(type='BatchCutMix', alpha=1.0, num_classes=2, prob=1.0)
   )
)

data = dict(
   samples_per_gpu=32,
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

# optimizer
optimizer = dict(type='HSPG', lr=0.01, lmbda=1e-3, momentum=0.9)
optimizer_config = dict(n_p=30, grad_clip=False, type='HSPGOptimizerHook')
optimizing_config = dict(init_stage='sgd', epsilon=[0.0, 0.0, 0.0, 0.0, 0.0], upper_group_sparsity=[1.0, 1.0, 1.0, 1.0, 1.0])
# learning policy
lr_config = dict(
   policy='CosineAnnealing',
   min_lr=0,
   warmup='linear',
   warmup_iters=5,
   warmup_ratio=0.1,
   warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=100)
