# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import math
import os
import sys

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MyModule(nn.Module):

    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

class MyNetwork(MyModule):
	CHANNEL_DIVISIBLE = 8

	def forward(self, x):
		raise NotImplementedError

	@property
	def module_str(self):
		raise NotImplementedError

	@property
	def config(self):
		raise NotImplementedError

	@staticmethod
	def build_from_config(config):
		raise NotImplementedError

	def zero_last_gamma(self):
		raise NotImplementedError

	@property
	def grouped_block_index(self):
		raise NotImplementedError

	""" implemented methods """

	def set_bn_param(self, momentum, eps, gn_channel_per_group=None, **kwargs):
		set_bn_param(self, momentum, eps, gn_channel_per_group, **kwargs)

	def get_bn_param(self):
		return get_bn_param(self)

	def get_parameters(self, keys=None, mode='include'):
		if keys is None:
			for name, param in self.named_parameters():
				if param.requires_grad: yield param
		elif mode == 'include':
			for name, param in self.named_parameters():
				flag = False
				for key in keys:
					if key in name:
						flag = True
						break
				if flag and param.requires_grad: yield param
		elif mode == 'exclude':
			for name, param in self.named_parameters():
				flag = True
				for key in keys:
					if key in name:
						flag = False
						break
				if flag and param.requires_grad: yield param
		else:
			raise ValueError('do not support: %s' % mode)

	def weight_parameters(self):
		return self.get_parameters()



def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'h_swish':
        return Hswish(inplace=inplace)
    elif act_func == 'h_sigmoid':
        return Hsigmoid(inplace=inplace)
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


class ShuffleLayer(nn.Module):

    def __init__(self, groups):
        super(ShuffleLayer, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    REDUCTION = 4

    def __init__(self, channel):
        super(SEModule, self).__init__()

        self.channel = channel
        self.reduction = SEModule.REDUCTION

        num_mid = make_divisible(self.channel // self.reduction, divisor=8)

        self.fc = nn.Sequential(OrderedDict([
            ('reduce', nn.Conv2d(self.channel, num_mid, 1, 1, 0, bias=True)),
            ('relu', nn.ReLU(inplace=True)),
            ('expand', nn.Conv2d(num_mid, self.channel, 1, 1, 0, bias=True)),
            ('h_sigmoid', Hsigmoid(inplace=True)),
        ]))

    def forward(self, x):
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.fc(y)
        return x * y

def min_divisible_value(n1, v1):
	""" make sure v1 is divisible by n1, otherwise decrease v1 """
	if v1 >= n1:
		return n1
	while n1 % v1 != 0:
		v1 -= 1
	return v1

def make_divisible(v, divisor, min_val=None):
	"""
	This function is taken from the original tf repo.
	It ensures that all layers have a channel number that is divisible by 8
	It can be seen here:
	https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
	:param v:
	:param divisor:
	:param min_val:
	:return:
	"""
	if min_val is None:
		min_val = divisor
	new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v


def set_bn_param(net, momentum, eps, gn_channel_per_group=None, ws_eps=None, **kwargs):
	replace_bn_with_gn(net, gn_channel_per_group)

	for m in net.modules():
		if type(m) in [nn.BatchNorm1d, nn.BatchNorm2d]:
			m.momentum = momentum
			m.eps = eps
		elif isinstance(m, nn.GroupNorm):
			m.eps = eps

	replace_conv2d_with_my_conv2d(net, ws_eps)
	return
    
def replace_bn_with_gn(model, gn_channel_per_group):
	if gn_channel_per_group is None:
		return

	for m in model.modules():
		to_replace_dict = {}
		for name, sub_m in m.named_children():
			if isinstance(sub_m, nn.BatchNorm2d):
				num_groups = sub_m.num_features // min_divisible_value(sub_m.num_features, gn_channel_per_group)
				gn_m = nn.GroupNorm(num_groups=num_groups, num_channels=sub_m.num_features, eps=sub_m.eps, affine=True)

				# load weight
				gn_m.weight.data.copy_(sub_m.weight.data)
				gn_m.bias.data.copy_(sub_m.bias.data)
				# load requires_grad
				gn_m.weight.requires_grad = sub_m.weight.requires_grad
				gn_m.bias.requires_grad = sub_m.bias.requires_grad

				to_replace_dict[name] = gn_m
		m._modules.update(to_replace_dict)

def replace_conv2d_with_my_conv2d(net, ws_eps=None):
	if ws_eps is None:
		return

	for m in net.modules():
		to_update_dict = {}
		for name, sub_module in m.named_children():
			if isinstance(sub_module, nn.Conv2d) and not sub_module.bias:
				# only replace conv2d layers that are followed by normalization layers (i.e., no bias)
				to_update_dict[name] = sub_module
		for name, sub_module in to_update_dict.items():
			m._modules[name] = MyConv2d(
				sub_module.in_channels, sub_module.out_channels, sub_module.kernel_size, sub_module.stride,
				sub_module.padding, sub_module.dilation, sub_module.groups, sub_module.bias,
			)
			# load weight
			m._modules[name].load_state_dict(sub_module.state_dict())
			# load requires_grad
			m._modules[name].weight.requires_grad = sub_module.weight.requires_grad
			if sub_module.bias is not None:
				m._modules[name].bias.requires_grad = sub_module.bias.requires_grad
	# set ws_eps
	for m in net.modules():
		if isinstance(m, MyConv2d):
			m.WS_EPS = ws_eps