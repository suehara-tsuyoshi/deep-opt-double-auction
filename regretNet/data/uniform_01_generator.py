from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from base.base_generator import BaseGenerator

class Generator(BaseGenerator):
    def __init__(self, config, mode, X = None, X_ADV = None, Y = None, Y_ADV = None):
        super(Generator, self).__init__(config, mode)
        self.build_generator(X = X, X_ADV = X_ADV, Y = Y, Y_ADV=Y_ADV)

    def generate_random_X(self, shape):
        return np.random.rand(*shape)
        
        # soboleng = torch.quasirandom.SobolEngine(dimension=shape[2])
        # X = torch.empty(shape)
        # for i in range(shape[0]):
        #     val = soboleng.draw(shape[1])
        #     X[i, :, :] = val
        # X_numpy = X.to('cpu').detach().numpy().copy()
        # return X_numpy

    def generate_random_ADV(self, shape):
        return np.random.rand(*shape)