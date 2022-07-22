from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


def adap_alpha(kdloss, gtloss):
    gamma = 2.0
    tloss = kdloss + gtloss
    norm_kdloss = kdloss / tloss
    norm_gtloss = gtloss / tloss
    alphakd = torch.pow(norm_kdloss, gamma) * 2.
    alphagt = torch.pow(norm_gtloss, gamma) * 2.
    alphakd = min(alphakd, 0.8)
    alphagt = max(alphagt, 0.2)
    return alphakd, alphagt


def loss_fn_kd(outputs, labels, teacher_outputs, opt):
    alpha = opt.alpha
    T = opt.temperature
    
    KD = F.kl_div(F.log_softmax(outputs/T, dim=1),
                    F.softmax(teacher_outputs/T, dim=1),
                    reduction='batchmean') * T * T * alpha
    
    GT_loss = F.cross_entropy(outputs, labels) * (1. - alpha) 
    KD_loss = KD + GT_loss
    return KD_loss

def loss_fn_kd_obj(outputs, labels, teacher_outputs, opt):
    alpha = opt.alpha
    T = opt.temperature

    return None