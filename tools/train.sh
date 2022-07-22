# python -m torch.distributed.launch --nproc_per_node=2 --master_port=2022 tools/train.py \
# configs/resnet/resnet18_b16x8_cifar10.py --pruned_model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/chek/prune_model/resnet18_b16x8_cifar10_sparsity_0.2.pth \
# --launcher pytorch

# python -m torch.distributed.launch --nproc_per_node=2 --master_port=2022 tools/train.py \
# configs/resnet/resnet34_8xb16_cifar10.py --pruned_model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/chek/prune_model/resnet34_8xb16_cifar10_sparsity_0.2.pth \
# --launcher pytorch

# python -m torch.distributed.launch --nproc_per_node=2 --master_port=2022 tools/train.py \
# configs/resnet/resnet50_8xb16_cifar10.py --pruned_model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/chek/prune_model/resnet50_8xb16_cifar10_sparsity_0.2.pth \
# --launcher pytorch

# python -m torch.distributed.launch --nproc_per_node=2 --master_port=2023 tools/train.py \
# configs/resnet/resnet101_8xb16_cifar10.py --pruned_model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/chek/prune_model/resnet101_8xb16_cifar10_sparsity_0.2.pth \
# --launcher pytorch

# python -m torch.distributed.launch --nproc_per_node=2 --master_port=2024 tools/train.py \
# configs/resnet/resnet152_8xb16_cifar10.py --pruned_model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/chek/prune_model/resnet152_8xb16_cifar10_sparsity_0.2.pth \
# --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=2 --master_port=2024 tools/train.py \
configs/mobilenet_v3/mobilenet-v3-small_8xb16_cifar10.py --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=2 --master_port=2025 tools/train.py \
configs/mobilenet_v2/mobilenet_v2_8xb16_cifar10.py --pruned_model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/chek/prune_model/mobilenet_v2_8xb16_cifar10_sparsity_0.2.pth \
--launcher pytorch

python -m torch.distributed.launch --nproc_per_node=2 --master_port=2024 tools/train.py \
configs/resnest/resnest50_8xb16_cifar10.py --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=2 --master_port=2024 tools/train.py \
configs/resnext/resnext50_32x4d_8xb16_cifar10.py --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=2 --master_port=2025 tools/train.py \
configs/fanet/fanet_b0_finetune_cifar10.py --launcher pytorch


python -m torch.distributed.launch --nproc_per_node=2 --master_port=2026 tools/train.py \
configs/repvgg/repvgg-A0_4xb64_cifar.py --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=2 --master_port=2025 tools/train.py \
configs/fanet/fanet_b0_finetune_cifar10.py --pruned_model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/chek/prune_model/fanet_b0_finetune_cifar10_sparsity_0.2.pth \
--launcher pytorch

python -m torch.distributed.launch --nproc_per_node=2 --master_port=2026 tools/train.py \
configs/resnext/resnext50_32x4d_8xb16_cifar10.py --pruned_model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/chek/prune_model/resnext50_32x4d_8xb16_cifar10_sparsity_0.3.pth \
--launcher pytorch

python -m torch.distributed.launch --nproc_per_node=2 --master_port=2026 tools/train.py \
configs/resnext/resnext50_32x4d_8xb16_cifar10.py --pruned_model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/chek/prune_model/resnext50_32x4d_8xb16_cifar10_sparsity_0.3.pth \
--launcher pytorch

python -m torch.distributed.launch --nproc_per_node=2 --master_port=2027 tools/train.py \
configs/densenet/densenet121_8xb16_cifar10.py --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=2 --master_port=2027 tools/train.py \
configs/densenet/densenet121_8xb16_cifar10.py --pruned_model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/chek/prune_model/densenet121_8xb16_cifar10_sparsity_0.2.pth \
--launcher pytorch


python -m torch.distributed.launch --nproc_per_node=2 --master_port=2026 tools/train.py \
configs/efficientnet/efficientnet_b0_8xb16_cifar10.py --launcher pytorch


python -m torch.distributed.launch --nproc_per_node=2 --master_port=2026 tools/train.py \
configs/seresnet/seresnet_8xb16_cifar10.py --launcher pytorch


python -m torch.distributed.launch --nproc_per_node=2 --master_port=2027 tools/train.py \
 configs/regnet/regnetx_1.6gf_b32x8_cifar10.py --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=2 --master_port=2027 tools/train.py \
configs/regnet/regnetx_1.6gf_b32x8_cifar10.py --pruned_model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/chek/prune_model/regnetx_1.6gf_b32x8_cifar10_sparsity_0.1.pth \
--launcher pytorch

python -m torch.distributed.launch --nproc_per_node=2 --master_port=2028 tools/train.py \
configs/seresnet/seresnet_8xb16_cifar10.py --pruned_model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/chek/prune_model/seresnet_8xb16_cifar10_sparsity_0.2.pth \
--launcher pytorch


python -m torch.distributed.launch --nproc_per_node=2 --master_port=2026 tools/train.py \
configs/automl_atum/pruning/L2Pruner_res50_xb256_cifa10_ratio0.5.py --launcher pytorch


python -m torch.distributed.launch --nproc_per_node=2 --master_port=2026 tools/automl_tools/train_mmcls.py \
configs/automl_atum/pruning/L2Pruner_res50_xb256_cifa10_ratio0.5_mx.py --launcher pytorch


python -m torch.distributed.launch --nproc_per_node=2 --master_port=2028 tools/train.py configs/faster_rcnn/faster_pruning.py --launch pytorch