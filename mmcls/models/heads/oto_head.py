# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class OTOClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super(OTOClsHead, self).__init__(*args, **kwargs)



    # def simple_test(self, x):
    #     """Test without augmentation."""
    #     if isinstance(x, tuple):
    #         x = x[-1]
    #     cls_score = self.fc(x)
    #     if isinstance(cls_score, list):
    #         cls_score = sum(cls_score) / float(len(cls_score))
    #     pred = F.softmax(cls_score, dim=1) if cls_score is not None else None

    #     return self.post_process(pred)

    def forward_train(self, x, gt_label, **kwargs):
        if isinstance(x, tuple):
            x = x[-1]
        cls_score = x
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses
