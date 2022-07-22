#
#  Modified by liujunjie on 2020/12/17.
#  Copyright (c) Meituan. Holding Limited
#  Email liujunjie10@meituan.com
#

#import argparse
#import ast

def quant_both_func():
    config_list = [{
            'quant_types': ['weight'],
            'quant_bits': {
                'weight': 8,
            }, 
            #'quant_start_step': 10,
            #'op_types':['Conv2d', 'Linear', 'GRU', 'LSTM', 'RNN'],
            'op_types':['Conv2d', 'GRU', 'LSTM', 'RNN', 'Linear'],
            'asymmetric': 0
        }, {
            'quant_types': ['output'],
            'quant_bits': 8,
            #'quant_start_step': 7000,
            #'op_types': ["None"],
            #'op_types':['ReLU', 'ReLU6', 'LSTM', 'RNN'],
            'op_types':['ReLU', 'ReLU6', 'LSTM', 'RNN', 'Linear'],
            #'op_types':['ReLU', 'ReLU6', 'GRU', 'LSTM', 'RNN', 'Linear'],
            'asymmetric': 0   
    }]
    return config_list

def quant_weight_func():
    config_list = [{
            'quant_types': ['weight'],
            'quant_bits': {
                'weight': 8,
            }, 
            #'quant_start_step': 10,
            'op_types':['Conv2d', 'Linear', 'GRU', 'LSTM', 'RNN'],
            'asymmetric': 0
    }]
    return config_list


