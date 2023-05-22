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
        num_buyers = self.config.num_buyers
        num_sellers = self.config.num_sellers
        num_agents = self.config.num_agents

        num_a_layers = self.config.net.num_a_layers
        num_p_layers = self.config.net.num_p_layers
        num_r_layers = self.config.net.num_r_layers

        num_a_hidden_units = self.config.net.num_a_hidden_units
        num_p_hidden_units = self.config.net.num_p_hidden_units
        num_r_hidden_units = self.config.net.num_r_hidden_units

        num_in = num_agents
        num_out = (num_buyers+1) * (num_sellers+1)
        
        # construct allocation model
        self.alloc_list = []
        self.alloc_list.append(nn.Linear(num_agents, num_a_hidden_units))
        self.alloc_list.append(self.activation)
        for i in range(1, num_a_layers - 1):
            self.alloc_list.append(nn.Linear(num_a_hidden_units, num_a_hidden_units))
            self.alloc_list.append(self.activation)
        # self.alloc_list.append(nn.Linear(num_a_hidden_units, num_out))
        self.alloc = nn.Sequential(*self.alloc_list)

        self.alloc_buyer_list = []
        self.alloc_buyer_list.append(nn.Linear(num_a_hidden_units, num_out))
        self.alloc_buyer = nn.Sequential(*self.alloc_buyer_list)

        self.alloc_seller_list = []
        self.alloc_seller_list.append(nn.Linear(num_a_hidden_units, num_out))
        self.alloc_seller = nn.Sequential(*self.alloc_seller_list)

        # construct payment model
        self.pay_list = []
        self.pay_list.append(nn.Linear(num_agents, num_p_hidden_units))
        self.pay_list.append(self.activation)
        for i in range(1, num_p_layers - 1):
            self.pay_list.append(nn.Linear(num_p_hidden_units, num_p_hidden_units))
            self.pay_list.append(self.activation)
        self.pay_list.append(nn.Linear(num_p_hidden_units, num_buyers))
        self.pay = nn.Sequential(*self.pay_list)

        # construct revenue model
        self.rev_list = []
        self.rev_list.append(nn.Linear(num_agents, num_r_hidden_units))
        self.rev_list.append(self.activation)
        for i in range(1, num_r_layers - 1):
            self.rev_list.append(nn.Linear(num_r_hidden_units, num_r_hidden_units))
            self.rev_list.append(self.activation)
        self.rev_list.append(nn.Linear(num_r_hidden_units, num_sellers))
        self.rev = nn.Sequential(*self.rev_list)

        self.weight_init(self.alloc)
        self.weight_init(self.alloc_buyer)
        self.weight_init(self.alloc_seller)
        self.weight_init(self.pay)
        self.weight_init(self.rev)
    
    def inference(self, x):

        b_x_in, s_x_in = torch.split(x, [self.config.num_buyers, self.config.num_sellers], dim=-1)
        sorted_b_x, sorted_b_x_idx = torch.sort(b_x_in, dim=-1, descending=True)
        sorted_s_x, sorted_s_x_idx = torch.sort(s_x_in, dim=-1)
        x_in = torch.cat((sorted_b_x, sorted_s_x), dim=-1) 

        x_in = x_in.reshape([-1, self.config.num_agents])
        x_out = self.alloc(x_in)
        
        buyer = self.alloc_buyer(x_out)
        buyer = buyer.reshape([-1, self.config.num_buyers + 1, self.config.num_sellers + 1])
        buyer = F.softmax(buyer, dim = 1)

        seller = self.alloc_seller(x_out)
        seller = seller.reshape([-1, self.config.num_buyers + 1, self.config.num_sellers + 1])
        seller = F.softmax(seller, dim = -1)

        a = torch.minimum(buyer, seller)
        a = a[:, :self.config.num_buyers, :self.config.num_sellers]

        b_repeat_shape = [1] * a.dim()
        b_repeat_shape[-1] = self.config.num_sellers
        s_repeat_shape = [1] * a.dim()
        s_repeat_shape[-1] = self.config.num_buyers
        
        sorted_b_x_idx = torch.argsort(sorted_b_x_idx)
        sorted_s_x_idx = torch.argsort(sorted_s_x_idx)

        sorted_b_x_idx_2d = sorted_b_x_idx.repeat(b_repeat_shape)
        sorted_s_x_idx_2d = sorted_s_x_idx.repeat(s_repeat_shape)
        
        sorted_b_x_idx_2d = sorted_b_x_idx_2d.reshape([-1, self.config.num_sellers, self.config.num_buyers])
        sorted_s_x_idx_2d = sorted_s_x_idx_2d.reshape([-1, self.config.num_buyers, self.config.num_sellers])
        sorted_b_x_idx_2d = sorted_b_x_idx_2d.transpose(-1, -2)
        
        a = torch.gather(a, -2, sorted_b_x_idx_2d)
        a = torch.gather(a, -1, sorted_s_x_idx_2d)

        
        p = self.pay(x_in)
        p = torch.sigmoid(p)
        p = torch.gather(p, -1, sorted_b_x_idx)

        b_u = (a.sum(-1) * b_x_in).reshape([-1, self.config.num_buyers])
        p = b_u * p


        r = self.rev(x_in)
        r = torch.sigmoid(r)
        r = torch.gather(r, -1, sorted_s_x_idx)

        s_u = (a.sum(-2) * s_x_in).reshape([-1, self.config.num_sellers])
        r = s_u / r
        
        return a, p, r
        


