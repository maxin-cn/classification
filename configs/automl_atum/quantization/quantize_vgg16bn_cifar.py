_base_ = [
    '../../_base_/models/vgg16bn.py',
    '../../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../../_base_/schedules/imagenet_bs256.py', '../../_base_/default_runtime.py'
]

data = dict(
    train=dict(data_prefix='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/cifar'),
    val=dict(data_prefix='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/cifar'),
    test=dict(data_prefix='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/cifar'),
)

model = dict(
    pretrained="/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/lee/Codebase/automl-atum/work_dirs/Res18_cifar/latest.pth"
)

# Quantize-Aware Training schedule config
# lr is set for a batch size of 128
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[15])
runner = dict(type='EpochBasedRunner', max_epochs=10)
log_config = dict(interval=100)