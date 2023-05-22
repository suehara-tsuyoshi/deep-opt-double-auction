from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import numpy as np
# import tensorflow as tf

import torch
from torch import nn
from torch.nn import functional as F


class BaseNet(object):
    
    def __init__(self, config):
        self.config = config

        """ Set initializer """
        if self.config.net.init == 'None':
            init = None
        elif self.config.net.init == 'gu':
            # tf.keras.initializers.glorot_uniform()
            init = lambda *x: nn.init.xavier_uniform_(*x)
        elif self.config.net.init == 'gn':
            # tf.keras.initializers.glorot_normal()
            init = lambda *x: nn.init.xavier_normal_(*x)
        elif self.config.net.init == 'hu':
            # tf.keras.initializers.he_uniform()
            init = lambda *x: nn.init.kaiming_uniform_(*x)
        elif self.config.net.init == 'hn':
            # tf.keras.initializers.he_normal()
            init = lambda *x: nn.init.kaiming_normal_(*x)
        self.init = init

        
        if self.config.net.activation == 'tanh': activation = nn.Tanh()
        elif self.config.net.activation == 'relu': activation = nn.ReLU()
        self.activation = activation    

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            self.init(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)    
               
    def build_net(self):
        """
        Initializes network variables
        """
        raise NotImplementedError
        
    def inference(self, x):
        """ 
        Inference 
        """
        raise NotImplementedError
        
            
            
