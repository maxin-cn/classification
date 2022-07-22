#
#  Modified by liujunjie on 2020/12/17.
#  Copyright (c) Meituan. Holding Limited
#  Email liujunjie10@meituan.com
#

import torch
import os
import shutil
import sys
import time
import os.path as osp
from torch.autograd import Variable


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('the loaded model size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    num = output.size(1)
    target_topk = []
    appendices = []
    for k in topk:
        if k <= num:
            target_topk.append(k)
        else:
            appendices.append([0.0])
    topk = target_topk
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res + appendices

"""
def accuracy(output, target, topk=(1,)):
    #Computes the accuracy@k for the specified values of k
    if len(topk) > 1:
        maxk = max(topk)
    else:
        maxk = topk[0]

    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
"""

def save_checkpoint(state, is_best, epoch, ppath, filename='checkpoint'):
    filename = filename + str(epoch) + 'pth.tar'
    if not osp.exists(ppath):
        os.mkdir(ppath)
    torch.save(state, osp.join(ppath, filename))
    if is_best:
        model_name = 'model_best_epoch_' + str(epoch) + ".pth.tar" 
        shutil.copyfile(osp.join(ppath, filename), osp.join(ppath, 'model_best.pth.tar'))


def norm_export_onnx(net, name, ifgpu):
    debugx = Variable(torch.randn(1, 3, 224, 224), requires_grad=True)
    debugx = debugx.cuda() if ifgpu else debugx.float()
    torch.onnx._export(net,                                        # model being run
            debugx,                                                # model input (or a tuple for multiple inputs)
            "./" + name + ".onnx",                                 # where to save the model (can be a file or file-like object)
            verbose=True, export_params=True, training=False)      # store the trained parameter weights inside the model file

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def accumulate(self, val, n=1):
        self.sum += val
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

