#
#  Modified by liujunjie on 2020/12/17.
#  Copyright (c) Meituan. Holding Limited
#  Email liujunjie10@meituan.com
#
import os
import torch
import torchvision
import numpy as np
import math
import time
from SKDX.algorithms.optimization.tools import AverageMeter, accuracy

best_acc    = 0
optimizer   = None
criterion   = None
FrozenBN    = 3
FrozenObser = 2

def grad_step(net, epoch, train_loader, device):
    net.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        precs = accuracy(outputs.data, targets.data, topk=(1, ))
        prec1 = precs[0]
        if torch.cuda.is_available():
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
        else:
            losses.update(loss, inputs.size(0))
            top1.update(prec1, inputs.size(0))      
        # timing
        batch_time.update(time.time() - end)
        end = time.time()

        print('Loss: {:.3f} | Acc1: {:.3f}%'.format(losses.avg, top1.avg))

def no_grad_step(net, epoch, test_loader, device, save=True):
    global best_acc
    net.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            precs = accuracy(outputs.data, targets.data, topk=(1, ))
            prec1 = precs[0]
            if torch.cuda.is_available():
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
            else:
                losses.update(loss, inputs.size(0))
                top1.update(prec1, inputs.size(0))  
            # timing
            batch_time.update(time.time() - end)
            end = time.time()

            print('Loss: {:.3f} | Acc1: {:.3f}%'.format(losses.avg, top1.avg))

        is_best = False
        if top1.avg > best_acc:
            best_acc = top1.avg
            is_best = True

        print('Current best acc: {}'.format(best_acc))

def adjust_learning_rate(optimizer, epoch, max_epoch, init_learning_rate, lr_type='cos'):
    if lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * init_learning_rate * (1 + math.cos(math.pi * epoch / max_epoch))
    elif lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = init_learning_rate * (decay ** (epoch // step))
    else:
        lr = init_learning_rate

    print('=> lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def fine_tuning(model, device, strategy='pruning', model_arch='mobilenetv2', train_loader=None, val_loader=None):
    global optimizer
    global criterion

    init_learning_rate = 0.05
    if model_arch == 'mobilenetv2':
        init_learning_rate = 0.01
    elif model_arch == 'resnet50':
        init_learning_rate = 0.045
    
    default_epoch      = 30
    if strategy == 'pruning':
        default_epoch  = 65
    elif strategy == 'qat':
        default_epoch  = 15
    elif strategy == 'debug':
        default_epoch  = 1

    optimizer = torch.optim.SGD(model.parameters(), lr=init_learning_rate, momentum=0.9, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    for epoch in range(0, default_epoch):
        if strategy == 'pruning':
            lr = adjust_learning_rate(optimizer, epoch, default_epoch, init_learning_rate)
        if strategy == 'qat':
            lr = adjust_learning_rate(optimizer, epoch, default_epoch, init_learning_rate)
            """
            if epoch > 2:
                freeze_obser
            if epoch > 3:
                freeze_bn
            """
        grad_step(model, epoch, train_loader, device)
        no_grad_step(model, epoch, val_loader, device)
        