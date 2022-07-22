custom_imports = dict(imports=['mmcls.core.optimizer.hspg'], allow_failed_imports=False)
custom_imports = dict(imports=['mmcls.core.utils.hspg_optimizer_hook'], allow_failed_imports=False)

_base_ = [
    '../_base_/models/resnet50_oto_imagenet.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256_oto.py', '../_base_/default_runtime.py'
]

optimizing_config = dict(init_stage='sgd', epsilon=[0.0, 0.0, 0.0, 0.0, 0.0], upper_group_sparsity=[1.0, 1.0, 1.0, 1.0, 1.0])
