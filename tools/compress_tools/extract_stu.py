import torch
import torchvision
import numpy 
import argparse
import logging
import sys
import os

parser = argparse.ArgumentParser(description='extracting the student model from KD pipeline.')
parser.add_argument('--ckpt', type=str, 
                    help='the ckpt from KD pipeline.')
parser.add_argument('--path_stu_ckpt', type=str,
                    help='the output path of stu ckpt for further test')

if __name__ == '__main__':
    opt = parser.parse_args()
    logging.info(opt)

    kd_dict = None
    stu_dict = {}    

    if opt.ckpt is None:
        logging.info("the ckpt from KD pipeline is missed")
    else:
        kd_dict = torch.load(opt.ckpt)
        kd_params = kd_dict['state_dict']
        for k, v in kd_params.items():
            if "architecture.model" in k:
                new_key = k.split('architecture.model.')[1]
                stu_dict[new_key] = v
        kd_dict['state_dict'] = stu_dict
        torch.save(kd_dict, opt.path_stu_ckpt)

