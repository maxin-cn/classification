#
#  Modified by liujunjie on 2020/12/17.
#  Copyright (c) Meituan. Holding Limited
#  Email liujunjie10@meituan.com
#

#import argparse
#import ast
import os

class abc_configs():
    def __init__(self, random_rule='default', calfitness_epoch=2, max_cycle=10, min_preserve=0.1,
                    max_preserve=0.9, preserve_type='layerwise', food_number=10, food_dimension=13, food_limit=5, honeychange_num=2):
        self.random_rule      = random_rule        # 'default','random_pretrain','l1_pretrain'
        self.calfitness_epoch = calfitness_epoch   # 
        self.max_cycle        = max_cycle          #
        self.min_preserve     = min_preserve       #
        self.max_preserve     = max_preserve       #
        self.preserve_type    = preserve_type      #
        self.food_number      = food_number        #
        self.food_dimension   = food_dimension     #
        self.food_limit       = food_limit         #
        self.honeychange_num  = honeychange_num    #

class amc_configs():
    def __init__(self, layer_types=['Conv2d', 'Linear'], episode=800, sparse_ratio=0.7, lower_bound=0.5, upper_bound=0.9, suffix=None):
        self.layer_types  = layer_types     # the type of layer to compress
        self.episode      = episode         # the step amounts to search
        self.sparse_ratio = sparse_ratio    # the targeted ratio to prune
        self.lower_bound  = lower_bound     # the lower bound for each layer to prune
        self.upper_bound  = upper_bound     # the upper bound for each layer to prune
        self.suffix       = suffix                    

