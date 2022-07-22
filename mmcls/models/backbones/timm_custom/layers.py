import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers.padding import get_padding, get_padding_value, pad_same


class StdConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """
    def __init__(
            self, in_channel, out_channels, kernel_size, stride=1, padding=None,
            dilation=1, groups=1, bias=False, eps=1e-6):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channel, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def forward(self, x):
        """ To ease onnx conversion,
        c.f. https://github.com/google-research/big_transfer/blob/49afe42338b62af9fbe18f0258197a33ee578a6b/bit_pytorch/models.py#L25 
        see also https://km.sankuai.com/page/1308630148
        """
        if self.training:
            self.running_mean = None
            self.running_var = None
            self.weight = F.batch_norm(
                self.weight.reshape(1, self.out_channels, -1), self.running_mean, self.running_var,
                    training=self.training, momentum=0., eps=self.eps).reshape_as(self.weight)
        
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x
