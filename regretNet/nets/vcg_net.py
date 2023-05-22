from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from mimetypes import init
from pyexpat import model

import numpy as np
# import tensorflow as tf

import torch
from torch import nn
from torch.nn import functional as F

from base.base_net import *

class Net(BaseNet):
    def __init__(self, config):
        super(Net, self).__init__(config)
        self.build_net()

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    def build_net(self):
        return
    
    def inference(self, x, y):

        # x_in = torch.chunk(x, 2, dim=0)
        buyer_x = x
        buyer_x = buyer_x.reshape([-1, self.config.num_buyers])
        seller_x = y
        seller_x = seller_x.reshape([-1, self.config.num_sellers])

        num_instances = buyer_x.shape[0]
        # a = torch.tensor([])
        # p = torch.tensor([])
        # r = torch.tensor([])
        a = []
        p = []
        r = []
        max_idx = min(self.config.num_buyers, self.config.num_sellers)
      

        for i in range(num_instances):
            alloc = [[0.0] * self.config.num_sellers for i in range(self.config.num_buyers)]
            pay = [0.0] * self.config.num_buyers
            rev = [0.0] * self.config.num_sellers
            buyer_x_sorted, buyer_x_idx = torch.sort(buyer_x[i], descending=True)
            seller_x_sorted, seller_x_idx = torch.sort(seller_x[i])

            idx = 0
    
            for j in range(max_idx):
                if buyer_x_sorted[j] >= seller_x_sorted[j]:
                    idx += 1
                else:
                    break
            
            if idx == max_idx:
                # price = (buyer_x_sorted[idx-1]+seller_x_sorted[idx-1])/2.0
                # for j in range(idx-1):
                #     alloc[buyer_x_idx[j]][seller_x_idx[j]] = 1.0
                #     pay[buyer_x_idx[j]] = buyer_x_sorted[idx-1]
                #     rev[seller_x_idx[j]] = seller_x_sorted[idx-1]

                price = 0.5

                for j in range(idx):
                        alloc[buyer_x_idx[j]][seller_x_idx[j]] = 1.0
                        pay[buyer_x_idx[j]] = max(seller_x_sorted[idx-1], 0.0)
                        rev[seller_x_idx[j]] = min(buyer_x_sorted[idx-1], 1.0)
                
                
                
            elif idx != 0:
                
                for j in range(idx):
                        alloc[buyer_x_idx[j]][seller_x_idx[j]] = 1.0
                        pay[buyer_x_idx[j]] = max(seller_x_sorted[idx-1], buyer_x_sorted[idx])
                        rev[seller_x_idx[j]] = min(buyer_x_sorted[idx-1], seller_x_sorted[idx])
            a.append(alloc)
            p.append(pay)
            r.append(rev)
        a = torch.tensor(a)
        p = torch.tensor(p)
        r = torch.tensor(r)
        # a = torch.cat(a, dim = 0)
        # p = torch.cat(p, dim = 0)
        # r = torch.cat(r, dim = 0)

        
        return a, p, r
        


