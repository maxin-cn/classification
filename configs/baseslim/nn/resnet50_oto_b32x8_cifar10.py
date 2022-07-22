custom_imports = dict(imports=['mmcls.core.optimizer.hspg'], allow_failed_imports=False)
custom_imports = dict(imports=['mmcls.core.utils.hspg_optimizer_hook'], allow_failed_imports=False)

_base_ = [
    '../../_base_/models/resnet50_oto_cifar.py', '../../_base_/datasets/cifar10_bs16.py',
    '../../_base_/schedules/cifar10_bs128_oto.py', '../../_base_/default_runtime.py'
]

optimizing_config = dict(init_stage='sgd', epsilon=[0.0, 0.0, 0.0, 0.0, 0.0], upper_group_sparsity=[1.0, 1.0, 1.0, 1.0, 1.0])
