import torch
from argparse import ArgumentParser

from mmcls.apis import inference_model, init_model, show_result_pyplot

from nni.compression.pytorch import ModelSpeedup
from nni.compression.pytorch.utils.counter import count_flops_params
from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import SlimPruner, L1NormPruner, FPGMPruner
from nni.compression.pytorch.utils import not_safe_to_prune

device = 'cuda:0'
# config = 'configs/repvgg/repvgg-A0_4xb64-coslr-120e_in1k.py'
# config = 'configs/resnet/resnet18_b16x8_cifar10.py'
# config = 'configs/resnet/resnet34_8xb16_cifar10.py'
# config = 'configs/resnet/resnet50_8xb16_cifar10.py'
# config = 'configs/resnet/resnet101_8xb16_cifar10.py'
# config = 'configs/resnet/resnet152_8xb16_cifar10.py'
# config = 'configs/mobilenet_v3/mobilenet_v3_small_cifar.py'
# config = 'configs/mobilenet_v2/mobilenet_v2_8xb16_cifar10.py'
# config = 'configs/resnext/resnext50_32x4d_8xb16_cifar10.py'
# config = 'configs/shufflenet_v2/shufflenet_v2_8xb16_cifar10.py'
# config = 'configs/densenet/densenet121_8xb16_cifar10.py'
# config = 'configs/efficientnet/efficientnet_b0_8xb16_cifar10.py' # xxxx
# config = 'configs/efficientnet/efficientnetv2_s_finetune_cifar.py'
# config = 'configs/res2net/res2net50-w14-s8_8xb32_cifar10.py'
config = 'configs/resnest/resnest50_8xb16_cifar10.py'
# config = 'configs/seresnet/seresnet_8xb16_cifar10.py'
# config = 'configs/fanet/fanet_b0_finetune_cifar10.py'
# config = 'configs/regnet/regnetx_1.6gf_b32x8_cifar10.py'
# checkpoint = 'chek/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth'
# checkpoint = 'chek/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth'
# checkpoint = 'chek/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth'
# checkpoint = 'chek/resnet/resnet101_b16x8_cifar10_20210528-2d29e936.pth'
# checkpoint = 'chek/resnet/resnet152_b16x8_cifar10_20210528-3e8e9178.pth'
# checkpoint = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/work_dirs/resnext50_32x4d_8xb16_cifar10/epoch_199.pth'
# checkpoint = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/work_dirs/resnext50_32x4d_8xb16_cifar10/epoch_199.pth'
# checkpoint = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/work_dirs/seresnet_8xb16_cifar10/epoch_199.pth'
# checkpoint = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/work_dirs/regnetx_1.6gf_b32x8_cifar10/epoch_199.pth'
checkpoint = None
img_file = 'demo/demo.JPEG'

# build the model from a config file and a checkpoint file
model = init_model(config, checkpoint, device=device)

model.forward = model.dummy_forward

pre_flops, pre_params, _ = count_flops_params(model, torch.randn([128, 3, 32, 32]).to(device))

im = torch.ones(1, 3, 128, 128).to(device)
out = model(im)
torch.jit.trace(model, im, strict=False)

# with torch.no_grad():
#     input_name = ['input']
#     output_name  = ['output']
#     onnxname = 'fanet.onnx'
#     torch.onnx.export(model, im, onnxname, input_names = input_name, output_names = output_name,
#                     opset_version=11, training=False, verbose=False, do_constant_folding=False)
#     print(f'successful export onnx {onnxname}')
# exit()

# scores = model(return_loss=False, **data)
# scores = model(return_loss=False, **im)

# test a single image
# result = inference_model(model, img_file)

# Start to prune and speedupls
print('\n' + '=' * 50 + ' START TO PRUNE THE BEST ACCURACY PRETRAINED MODEL ' + '=' * 50)
not_safe = not_safe_to_prune(model, im)



print('\n' + '=' * 50 +  'not_safe' + '=' * 50, not_safe)
cfg_list = []
for name, module in model.named_modules():
    print(name)
    if name in not_safe:
        continue
    if isinstance(module, torch.nn.Conv2d):
        cfg_list.append({'op_types':['Conv2d'], 'sparsity':0.2, 'op_names':[name]})

print('cfg_list')
for i in cfg_list:
    print(i)

pruner = FPGMPruner(model, cfg_list)
_, masks = pruner.compress()
pruner.show_pruned_weights()
pruner._unwrap_model()
pruner.show_pruned_weights()

from nni.compression.pytorch.speedup.compress_modules import no_replace
customized_replace_func = {
    'RSoftmax': lambda module, masks: no_replace(module, masks)
}

ModelSpeedup(model, dummy_input=im, masks_file=masks, confidence=32, customized_replace_func=customized_replace_func).speedup_model()
torch.jit.trace(model, im, strict=False)
print(model)
flops, params, results = count_flops_params(model, torch.randn([128, 3, 32, 32]).to(device))
print(f'Pretrained model FLOPs {pre_flops/1e6:.2f} M, #Params: {pre_params/1e6:.2f}M')
print(f'Finetuned model FLOPs {flops/1e6:.2f} M, #Params: {params/1e6:.2f}M')
model.forward = model.forward_
torch.save(model, 'chek/prune_model/res2net50-w14-s8_8xb32_cifar10_sparsity_0.2.pth')