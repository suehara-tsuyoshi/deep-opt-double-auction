from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np


class BaseGenerator(object):
    def __init__(self, config, mode = "train"):    
        self.config = config
        self.mode = mode
        self.num_items = config.num_items
        self.num_instances = config[self.mode].num_batches * config[self.mode].batch_size
        self.num_misreports = config[self.mode].num_misreports
        self.batch_size = config[self.mode].batch_size

        self.num_buyers = config.num_buyers
        self.num_sellers = config.num_sellers
                       
    def build_generator(self, X = None, X_ADV = None, Y = None, Y_ADV = None):
        if self.mode is "train":            
            if self.config.train.data is "fixed":
                if self.config.train.restore_iter == 0:
                    self.get_data(X, X_ADV, Y, Y_ADV)
                else:
                    self.load_data_from_file(self.config.train.restore_iter)
                self.gen_func = self.gen_fixed()
            else:
                self.gen_func = self.gen_online()
                
        else:
            if self.config[self.mode].data is "fixed" or X is not None:
                self.get_data(X, X_ADV, Y, Y_ADV)
                self.gen_func = self.gen_fixed()
            else:
                self.gen_func = self.gen_online()
            

        
    def get_data(self, X = None, X_ADV = None, Y = None, Y_ADV = None):
        """ Generates data """
        x_shape = [self.num_instances, self.num_buyers]
        x_adv_shape = [self.num_misreports, self.num_instances, self.num_buyers]
        y_shape = [self.num_instances, self.num_sellers]
        y_adv_shape = [self.num_misreports, self.num_instances, self.num_sellers]
        
        if X is None: X = self.generate_random_X(x_shape)
        if X_ADV is None: X_ADV = self.generate_random_ADV(x_adv_shape)
        if Y is None: Y = self.generate_random_X(y_shape)
        if Y_ADV is None: Y_ADV = self.generate_random_ADV(y_adv_shape)
            
        self.X = X
        self.X_ADV = X_ADV
        self.Y = Y
        self.Y_ADV = Y_ADV
                       
    def load_data_from_file(self, iter):
        """ Loads data from disk """
        self.X = np.load(os.path.join(self.config.dir_name, 'X.npy'))
        self.X_ADV = np.load(os.path.join(self.config.dir_name,'X_ADV_' + str(iter) + '.npy'))
        self.Y = np.load(os.path.join(self.config.dir_name, 'Y.npy'))
        self.Y_ADV = np.load(os.path.join(self.config.dir_name,'Y_ADV_' + str(iter) + '.npy'))
        
    def save_data(self, iter):
        """ Saved data to disk """
        if self.config.save_data is False: return
        
        if iter == 0:
            np.save(os.path.join(self.config.dir_name, 'X'), self.X)
            np.save(os.path.join(self.config.dir_name, 'Y'), self.Y)
        else:
            np.save(os.path.join(self.config.dir_name,'X_ADV_' + str(iter)), self.X_ADV)            
            np.save(os.path.join(self.config.dir_name,'Y_ADV_' + str(iter)), self.Y_ADV)            
                       
    def gen_fixed(self):
        i = 0
        if self.mode is "train": perm = np.random.permutation(self.num_instances) 
        else: perm = np.arange(self.num_instances)
            
        while True:
            idx = perm[i * self.batch_size: (i + 1) * self.batch_size]
            yield self.X[idx], self.X_ADV[:, idx, :], self.Y[idx], self.Y_ADV[:, idx, :], idx
            i += 1
            if(i * self.batch_size == self.num_instances):
                i = 0
                if self.mode is "train": perm = np.random.permutation(self.num_instances) 
                else: perm = np.arange(self.num_instances)
            
    def gen_online(self):
        x_batch_shape = [self.batch_size, self.num_buyers]
        x_adv_batch_shape = [self.num_misreports, self.batch_size, self.num_buyers]
        y_batch_shape = [self.batch_size, self.num_sellers]
        y_adv_batch_shape = [self.num_misreports, self.batch_size, self.num_sellers]
        while True:
            X = self.generate_random_X(x_batch_shape)
            X_ADV = self.generate_random_ADV(x_adv_batch_shape)
            Y = self.generate_random_X(y_batch_shape)
            Y_ADV = self.generate_random_ADV(y_adv_batch_shape)
            yield X, X_ADV, Y, Y_ADV, None

    def generate_random_X(self, shape):
        """ Rewrite this for new distributions """
        raise NotImplementedError

    def generate_random_ADV(self, shape):
        """ Rewrite this for new distributions """
        raise NotImplementedError