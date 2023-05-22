from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from email import generator

import os
import sys
import time
import logging
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, config, mode, net, clip_op_lambda):
        self.config = config
        self.mode = mode
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Create output-dir
        if not os.path.exists(self.config.dir_name):
            os.mkdir(self.config.dir_name)

        if self.mode == "train":
            log_suffix = '_' + \
                str(self.config.train.restore_iter) if self.config.train.restore_iter > 0 else ''
            self.log_fname = os.path.join(
                self.config.dir_name, 'train' + log_suffix + '.txt')
        else:
            log_suffix = "_iter_" + str(self.config.test.restore_iter) + "_m_" + str(
                self.config.test.num_misreports) + "_gd_" + str(self.config.test.gd_iter)
            self.log_fname = os.path.join(
                self.config.dir_name, "test" + log_suffix + ".txt")

        np.random.seed(self.config[self.mode].seed)

        self.init_logger()

        self.net = net
        self.net.alloc = self.net.alloc.to(self.device)
        self.net.alloc_buyer = self.net.alloc_buyer.to(self.device)
        self.net.alloc_seller = self.net.alloc_seller.to(self.device)
        self.net.pay = self.net.pay.to(self.device)
        self.net.rev = self.net.rev.to(self.device)

        self.clip_op_lambda = clip_op_lambda

        self.init_param()
    
    def init_logger(self):

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        handler = logging.FileHandler(self.log_fname, 'w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        self.logger = logger

    def compute_rev(self, pay):
        return torch.mean(pay.sum(-1))

    def compute_welf(self, b_x, s_x, alloc):
        welfare = torch.mean((torch.multiply(b_x, alloc.sum(-1))).sum(-1) - (torch.multiply(s_x, alloc.sum(-2))).sum(-1))
        return welfare
    
    def compute_utility(self, b_x, s_x, alloc, pay, revenue):
        buyer_utility = torch.multiply(b_x, alloc.sum(-1)) - pay
        seller_utility = revenue - torch.multiply(s_x, alloc.sum(-2))
        auctioneer_utility = pay.sum(-1)-revenue.sum(-1)
        return buyer_utility, seller_utility, auctioneer_utility
    
    def compute_entropy(self, alloc):

        unmatched_buyer_alloc = torch.ones([self.config[self.mode].batch_size, self.config.num_buyers])
        unmatched_seller_alloc = torch.ones([self.config[self.mode].batch_size, self.config.num_sellers])

        unmatched_buyer_alloc = unmatched_buyer_alloc.to(self.device)
        unmatched_seller_alloc = unmatched_seller_alloc.to(self.device)

        unmatched_buyer_alloc = unmatched_buyer_alloc - alloc.sum(-1)
        unmatched_seller_alloc = unmatched_seller_alloc - alloc.sum(-2)

        eps = 1e-6

        alloc = alloc * torch.log2(alloc + eps)

        unmatched_buyer_alloc = unmatched_buyer_alloc * torch.log2(unmatched_buyer_alloc + eps)
        unmatched_seller_alloc = unmatched_seller_alloc * torch.log2(unmatched_seller_alloc + eps)
        
        if self.config.num_agents != 2:
            buyer_entropy = - (alloc.sum(-1) + unmatched_buyer_alloc) / (2 * self.config.num_buyers * torch.log2(torch.tensor(self.config.num_sellers)))
            seller_entropy = - (alloc.sum(-2) + unmatched_seller_alloc) / (2 * self.config.num_sellers * torch.log2(torch.tensor(self.config.num_buyers)))
        else:
            buyer_entropy = - (alloc.sum(-1) + unmatched_buyer_alloc) 
            seller_entropy = - (alloc.sum(-2) + unmatched_seller_alloc)

        return buyer_entropy + seller_entropy

        

    
    def get_misreports(self, b_x, b_adv_var, s_x, s_adv_var):
        num_misreports = b_adv_var.shape[0]
        b_adv = b_adv_var.unsqueeze(0).tile([self.config.num_agents, 1, 1, 1])
        s_adv = s_adv_var.unsqueeze(0).tile([self.config.num_agents, 1, 1, 1])
        b_x_mis = b_x.tile([self.config.num_agents * num_misreports, 1])
        s_x_mis = s_x.tile([self.config.num_agents * num_misreports, 1])
        b_x_r = b_x_mis.reshape(self.b_adv_shape)
        s_x_r = s_x_mis.reshape(self.s_adv_shape)
        b_y = b_x_r * (1 - self.b_adv_mask) + b_adv * self.b_adv_mask
        s_y = s_x_r * (1 - self.s_adv_mask) + s_adv * self.s_adv_mask
        b_misreports = (b_y.reshape([-1, self.config.num_buyers])).float()
        s_misreports = (s_y.reshape([-1, self.config.num_sellers])).float()
        
        return b_x_mis, b_misreports, s_x_mis, s_misreports
    
    def init_param(self):

        self.b_x_shape = [self.config[self.mode].batch_size, self.config.num_buyers]
        self.s_x_shape = [self.config[self.mode].batch_size, self.config.num_sellers]
       
        self.b_adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_buyers]
        self.s_adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_sellers]
        self.adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]
        self.b_adv_var_shape = [ self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_buyers]
        self.s_adv_var_shape = [ self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_sellers]
        
        self.b_u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_buyers]
        self.s_u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_sellers]
        

        self.b_adv_mask = np.zeros(self.b_adv_shape)
        self.b_adv_mask[np.arange(self.config.num_buyers), :, :, np.arange(self.config.num_buyers)] = 1.0
        self.b_adv_mask = torch.from_numpy(self.b_adv_mask)

        self.s_adv_mask = np.zeros(self.s_adv_shape)
        self.s_adv_mask[np.arange(self.config.num_buyers, self.config.num_agents), :, :, np.arange(self.config.num_sellers)] = 1.0
        self.s_adv_mask = torch.from_numpy(self.s_adv_mask)
        

        self.adv_mask = np.zeros(self.adv_shape)
        self.adv_mask[np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents)] = 1.0
        self.adv_mask = torch.from_numpy(self.adv_mask)

        self.b_u_mask = np.zeros(self.b_u_shape)
        self.b_u_mask[np.arange(self.config.num_buyers), :, :, np.arange(self.config.num_buyers)] = 1.0
        self.b_u_mask = torch.tensor(self.b_u_mask, requires_grad=False)

        self.s_u_mask = np.zeros(self.s_u_shape)
        self.s_u_mask[np.arange(self.config.num_buyers, self.config.num_agents), :, :, np.arange(self.config.num_sellers)] = 1.0
        self.s_u_mask = torch.tensor(self.s_u_mask, requires_grad=False)

        self.b_adv_mask = self.b_adv_mask.to(self.device)
        self.s_adv_mask = self.s_adv_mask.to(self.device)
        self.b_u_mask = self.b_u_mask.to(self.device)
        self.s_u_mask = self.s_u_mask.to(self.device)
        
        if self.mode == "train":

            b_w_rgt_init_val = 0.0 if "b_w_rgt_init_val" not in self.config.train else self.config.train.b_w_rgt_init_val
            self.b_w_rgt = torch.tensor([b_w_rgt_init_val] * self.config.num_buyers, device = self.device, requires_grad=True)

            s_w_rgt_init_val = 0.0 if "s_w_rgt_init_val" not in self.config.train else self.config.train.s_w_rgt_init_val
            self.s_w_rgt = torch.tensor([s_w_rgt_init_val] * self.config.num_sellers, device = self.device, requires_grad=True)

            b_w_irp_init_val = 0.0 if "b_w_irp_init_val" not in self.config.train else self.config.train.b_w_irp_init_val
            self.b_w_irp = torch.tensor([b_w_irp_init_val] * self.config.num_buyers, device = self.device, requires_grad=True)

            s_w_irp_init_val = 0.0 if "s_w_irp_init_val" not in self.config.train else self.config.train.s_w_irp_init_val
            self.s_w_irp = torch.tensor([s_w_irp_init_val] * self.config.num_sellers, device = self.device, requires_grad=True)

            a_w_irp_init_val = 0.0 if "a_w_irp_init_val" not in self.config.train else self.config.train.a_w_irp_init_val
            self.a_w_irp = torch.tensor(a_w_irp_init_val, device = self.device, requires_grad=True)

            self.update_rate = self.config.train.update_rate
            self.learning_rate = self.config.train.learning_rate

            var_list = list(self.net.pay.parameters()) + list(self.net.rev.parameters())
            
            # Optimizer
            wd = None if "wd" not in self.config.train else self.config.train.wd
            if wd == None:
                wd = 0.0
            self.opt_1 = torch.optim.Adam(var_list, lr=self.learning_rate, weight_decay=wd)
            self.opt_3 = torch.optim.SGD([self.b_w_rgt, self.s_w_rgt], lr=self.update_rate, weight_decay=wd)

            # Metrics
            self.metric_names = ["Revenue", "Welfare", "Buyer_Regret", "Seller_Regret", "Reg_Loss", "Lag_Loss", "Net_Loss", "b_w_rgt", "s_w_rgt", "b_w_irp", "s_w_irp", "a_w_irp", "update_rate"]
            
        elif self.mode == "test":
            '''
            loss = -tf.reduce_sum(u_mis)
            test_mis_opt = tf.train.AdamOptimizer(self.config.test.gd_lr)
            self.test_mis_step = test_mis_opt.minimize(loss, var_list = [self.adv_var])
            self.reset_test_mis_opt = tf.variables_initializer(test_mis_opt.variables())

            # Metrics
            welfare = tf.reduce_mean(tf.reduce_sum(self.alloc * self.x, axis = (1,2)))
            self.metrics = [revenue, rgt_mean, irp_mean]
            '''
            self.metric_names = ["Revenue", "Welfare", "Buyer_Regret", "Seller_Regret", "Regret", "Buyer_IRP", "Seller_IRP", "Auctioneer_IRP", "Entropy"]

    def forward(self, X, X_ADV, Y, Y_ADV):

        x_mis, x_misreports, y_mis, y_misreports = self.get_misreports(X, X_ADV, Y, Y_ADV)

        self.alloc, self.pay, self.rev = self.net.inference(torch.cat((X, Y), dim=-1))
    
        a_mis, p_mis, r_mis = self.net.inference(torch.cat((x_misreports, y_misreports), dim=-1))

        b_utility, s_utility, a_utility = self.compute_utility(X, Y, self.alloc, self.pay, self.rev)
        b_utility_mis, s_utility_mis, a_utility_mis = self.compute_utility(x_mis, y_mis, a_mis, p_mis, r_mis)

        
        b_u_mis = b_utility_mis.reshape(self.b_u_shape) * self.b_u_mask
        s_u_mis = s_utility_mis.reshape(self.s_u_shape) * self.s_u_mask


        b_utility_true = b_utility.tile([self.config.num_agents * self.config[self.mode].num_misreports, 1])
        s_utility_true = s_utility.tile([self.config.num_agents * self.config[self.mode].num_misreports, 1])

        b_excess_from_utility = F.relu((b_utility_mis - b_utility_true).reshape(self.b_u_shape) * self.b_u_mask)
        s_excess_from_utility = F.relu((s_utility_mis - s_utility_true).reshape(self.s_u_shape) * self.s_u_mask)
        
        b_rgt = b_excess_from_utility.amax(dim=3).amax(1).mean(1)
        s_rgt = s_excess_from_utility.amax(dim=3).amax(1).mean(1)
        rgt = b_rgt + s_rgt
        
        revenue = self.compute_rev(self.pay)
        welfare = self.compute_welf(X, Y, self.alloc)
    
        b_irp = F.relu(-b_utility).mean(0)
        s_irp = F.relu(-s_utility).mean(0)
        a_irp = F.relu(-a_utility).mean(0)
        a_rev = -a_utility.mean(0)

        b_rgt = b_rgt[:self.config.num_buyers]
        s_rgt = s_rgt[self.config.num_buyers:]

        b_rgt = b_rgt.to(self.device)
        s_rgt = s_rgt.to(self.device)
        b_irp = b_irp.to(self.device)
        s_irp = s_irp.to(self.device)
        a_irp = a_irp.to(self.device)
        a_rev = a_rev.to(self.device)

        b_rgt_mean = b_rgt.mean()
        s_rgt_mean = s_rgt.mean()
        rgt_mean = rgt.mean()
        

        b_irp_mean = F.relu(-b_utility).mean()
        s_irp_mean = F.relu(-s_utility).mean()
        a_irp_mean = F.relu(-a_utility).mean()

        if self.mode == 'train':

            b_rgt_penalty = (self.update_rate) * (b_rgt * b_rgt).sum()/2.0
            s_rgt_penalty = (self.update_rate) * (s_rgt * s_rgt).sum()/2.0
            rgt_penalty = b_rgt_penalty + s_rgt_penalty

            irp_penalty = (self.update_rate) * (a_irp * a_irp)/2.0

            lag_loss = (self.b_w_rgt * b_rgt).sum() + \
                       (self.s_w_rgt * s_rgt).sum()     

            # loss_1 = -a_rev + rgt_penalty + lag_loss
            loss_1 = b_rgt.sum() + s_rgt.sum()
            # loss_2 = -b_u_mis.sum() -s_u_mis.sum()
            loss_2 = -b_u_mis.sum() -s_u_mis.sum()
            loss_3 = -lag_loss



            self.metrics = [revenue, welfare, b_rgt_mean, s_rgt_mean, rgt_penalty,
                            lag_loss, loss_1, self.b_w_rgt.mean(), self.s_w_rgt.mean(), self.b_w_irp.mean(), self.s_w_irp.mean(), self.a_w_irp.mean(), self.update_rate]
            
                    
        if self.mode == 'test':

            entropy = self.compute_entropy(self.alloc)
            entropy_mean = entropy.mean()

            loss_1 = 0
            loss_2 = -b_u_mis.sum() - s_u_mis.sum()
            loss_3 = 0

            self.metrics = [revenue, welfare, b_rgt_mean, s_rgt_mean, rgt_mean, b_irp_mean, s_irp_mean, a_irp_mean, entropy_mean]


        return loss_1, loss_2, loss_3

    def train(self, generator):

        self.train_gen, self.val_gen = generator

        # iter = self.config.train.restore_iter
        iter = 400000

        if iter > 0:
            model_path = os.path.join(
                self.config.dir_name, 'model-'+str(iter)+'.pt')
            
            checkpoint = torch.load(model_path)
            self.net.alloc.load_state_dict(
                checkpoint['model_alloc_state_dict'])
            self.net.alloc_buyer.load_state_dict(checkpoint['model_alloc_buyer_state_dict'])
            self.net.alloc_seller.load_state_dict(checkpoint['model_alloc_seller_state_dict'])
            self.net.pay.load_state_dict(checkpoint['model_pay_state_dict'])
            self.net.rev.load_state_dict(checkpoint['model_rev_state_dict'])
        
        if iter == 0:
            self.train_gen.save_data(0)
            torch.save({
                'epoch': iter,
                'model_alloc_state_dict': self.net.alloc.state_dict(),
                'model_alloc_buyer_state_dict': self.net.alloc_buyer.state_dict(),
                'model_alloc_seller_state_dict': self.net.alloc_seller.state_dict(),
                'model_pay_state_dict': self.net.pay.state_dict(),
                'model_rev_state_dict': self.net.rev.state_dict(),
                
            }, os.path.join(self.config.dir_name, 'model-0.pt'))
        
        iter = 0
        time_elapsed = 0.0
        while iter < (self.config.train.max_iter):

            X, X_ADV, Y, Y_ADV, perm = next(self.train_gen.gen_func)
            X = torch.from_numpy(X.astype(np.float32))
            X_ADV = torch.from_numpy(X_ADV.astype(np.float32))
            Y = torch.from_numpy(Y.astype(np.float32))
            Y_ADV = torch.from_numpy(Y_ADV.astype(np.float32))
            X = X.to(self.device)
            X_ADV = X_ADV.to(self.device)
            Y = Y.to(self.device)
            Y_ADV = Y_ADV.to(self.device)
            X_ADV.requires_grad_(True)
            Y_ADV.requires_grad_(True)
            
            # if iter == 0:
            #     self.opt_3.zero_grad()
            #     loss_1, loss_2, loss_3 = self.forward(X, X_ADV, Y, Y_ADV)
            #     loss_3.backward()
            #     self.opt_3.step()
            
            tic = time.time()

            wd = None if "wd" not in self.config.train else self.config.train.wd
            if wd == None:
                wd = 0.0
            self.opt_2 = torch.optim.Adam([X_ADV, Y_ADV], lr=self.config.train.gd_lr, weight_decay=wd)

            for _ in range(self.config.train.gd_iter):
                self.opt_2.zero_grad()
                loss_1, loss_2, loss_3 = self.forward(X, X_ADV, Y, Y_ADV)
                loss_2.backward()
                self.opt_2.step()
                X_ADV.requires_grad_(False)
                Y_ADV.requires_grad_(False)
                X_ADV = self.clip_op_lambda(X_ADV)
                Y_ADV = self.clip_op_lambda(Y_ADV)
                X_ADV.requires_grad_(True)
                Y_ADV.requires_grad_(True)
            
            self.opt_1.zero_grad()
            loss_1, loss_2, loss_3 = self.forward(X, X_ADV, Y, Y_ADV)
            loss_1.backward()
            self.opt_1.step()

            iter += 1

            # if iter % self.config.train.update_frequency == 0:
            #     self.opt_3.zero_grad()
            #     loss_1, loss_2, loss_3 = self.forward(X, X_ADV, Y, Y_ADV)
            #     loss_3.backward()
            #     self.opt_3.step()
            
            if iter % self.config.train.up_op_frequency == 0:
                self.update_rate = self.update_rate + self.config.train.up_op_add
            
            toc = time.time()
            time_elapsed += (toc - tic)

            if ((iter % self.config.train.save_iter) == 0) or (iter == self.config.train.max_iter):   
                torch.save({
                    'epoch': iter,
                    'model_alloc_state_dict': self.net.alloc.state_dict(),
                    'model_alloc_buyer_state_dict': self.net.alloc_buyer.state_dict(),
                    'model_alloc_seller_state_dict': self.net.alloc_seller.state_dict(),
                    'model_pay_state_dict': self.net.pay.state_dict(),
                    'model_rev_state_dict': self.net.rev.state_dict(),
                }, os.path.join(self.config.dir_name, 'remodel-'+str(iter)+'.pt'))
                self.train_gen.save_data(iter)

            if (iter % self.config.train.print_iter) == 0:
                # Train Set Stats
                self.forward(X, X_ADV, Y, Y_ADV)
                metric_vals = self.metrics
                fmt_vals = tuple([item for tup in zip(
                    self.metric_names, metric_vals) for item in tup])
                log_str = "TRAIN-BATCH Iter: %d, t = %.4f" % (
                    iter, time_elapsed) + ", %s: %.6f"*len(self.metric_names) % fmt_vals
                self.logger.info(log_str)

                

            if (iter % self.config.val.print_iter) == 0:
                # Validation Set Stats
                metric_tot = np.zeros(len(self.metric_names))
                for _ in range(self.config.val.num_batches):
                    X, X_ADV, Y, Y_ADV, _ = next(self.val_gen.gen_func)
                    X = torch.from_numpy(X.astype(np.float32))
                    X_ADV = torch.from_numpy(X_ADV.astype(np.float32))
                    Y = torch.from_numpy(Y.astype(np.float32))
                    Y_ADV = torch.from_numpy(Y_ADV.astype(np.float32))
                    X = X.to(self.device)
                    X_ADV = X_ADV.to(self.device)
                    Y = Y.to(self.device)
                    Y_ADV = Y_ADV.to(self.device)
                    X_ADV.requires_grad_(True)
                    
                    wd = None if "wd" not in self.config.train else self.config.train.wd
                    if wd == None:
                        wd = 0.0
                    self.val_mis_opt = torch.optim.Adam(
                        [X_ADV, Y_ADV], lr=self.config.val.gd_lr, weight_decay=wd)
                    for k in range(self.config.val.gd_iter):
                        self.val_mis_opt.zero_grad()
                        loss_1, loss_2, loss_3 = self.forward(X, X_ADV, Y, Y_ADV)
                        loss_2.backward()
                        self.val_mis_opt.step()
                        X_ADV.requires_grad_(False)
                        Y_ADV.requires_grad_(False)
                        self.clip_op_lambda(X_ADV)
                        self.clip_op_lambda(Y_ADV)
                        X_ADV.requires_grad_(True)
                        Y_ADV.requires_grad_(True)
                        
                    loss_1, loss_2, loss_3 = self.forward(X, X_ADV, Y, Y_ADV)

                    metric_vals = self.metrics
                    for i, v in enumerate(metric_vals):
                        metric_tot[i] += v

                metric_tot = metric_tot/self.config.val.num_batches
                fmt_vals = tuple([item for tup in zip(
                    self.metric_names, metric_tot) for item in tup])
                log_str = "VAL-%d" % (iter) + ", %s: %.6f" * \
                    len(self.metric_names) % fmt_vals
                self.logger.info(log_str)

                
    def test(self, generator):
        """
        Runs test
        """
        
        # Init generators
        self.test_gen = generator

        iter = self.config.test.restore_iter
        
        model_path = os.path.join(
            self.config.dir_name, 'remodel-'+str(iter)+'.pt')
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.net.alloc.load_state_dict(
            checkpoint['model_alloc_state_dict'])
        self.net.alloc_buyer.load_state_dict(checkpoint['model_alloc_buyer_state_dict'])
        self.net.alloc_seller.load_state_dict(checkpoint['model_alloc_seller_state_dict'])
        self.net.pay.load_state_dict(checkpoint['model_pay_state_dict'])
        self.net.rev.load_state_dict(checkpoint['model_rev_state_dict'])

        #Test-set Stats
        time_elapsed = 0
            
        metric_tot = np.zeros(len(self.metric_names))

        if self.config.test.save_output:
            assert(hasattr(generator, "X")), "save_output option only allowed when config.test.data = Fixed or when X is passed as an argument to the generator"
            alloc_tst = np.zeros([self.test_gen.X.shape[0], self.config.num_buyers, self.config.num_sellers])
            pay_tst = np.zeros([self.test_gen.X.shape[0], self.config.num_buyers])
            rev_tst = np.zeros([self.test_gen.X.shape[0], self.config.num_sellers])
                    
        for i in range(self.config.test.num_batches):
            tic = time.time()
            X, X_ADV, Y, Y_ADV, perm = next(self.test_gen.gen_func)
            X = torch.from_numpy(X.astype(np.float32))
            X_ADV = torch.from_numpy(X_ADV.astype(np.float32))
            Y= torch.from_numpy(Y.astype(np.float32))
            Y_ADV = torch.from_numpy(Y_ADV.astype(np.float32))
            X = X.to(self.device)
            Y = Y.to(self.device)
            X_ADV = X_ADV.to(self.device)
            Y_ADV = Y_ADV.to(self.device)
            X_ADV.requires_grad_(True)
            Y_ADV.requires_grad_(True)
                    
            wd = None if "wd" not in self.config.train else self.config.train.wd
            if wd == None:
                wd = 0.0
            self.opt_2 = torch.optim.Adam(
                [X_ADV, Y_ADV], lr=self.config.test.gd_lr, weight_decay=wd)
            
        
            for _ in range(self.config.test.gd_iter):
                self.opt_2.zero_grad()
                loss_1, loss_2, loss_3 = self.forward(X, X_ADV, Y, Y_ADV)
                loss_2.backward()
                self.opt_2.step()
                X_ADV.requires_grad_(False)
                Y_ADV.requires_grad_(False)
                self.clip_op_lambda(X_ADV) 
                self.clip_op_lambda(Y_ADV) 
                X_ADV.requires_grad_(True)
                Y_ADV.requires_grad_(True)
                
            
            
            if self.config.test.save_output:
                
                self.forward(X, X_ADV, Y, Y_ADV)
                alloc_tst[perm, :, :] = self.alloc.detach().cpu().numpy()
                pay_tst[perm, :] = self.pay.detach().cpu().numpy()
                rev_tst[perm, :] = self.rev.detach().cpu().numpy()
                # print(self.alloc)
                # print(self.pay)
                # print(self.rev)
            
            metric_vals = self.metrics
            for i, v in enumerate(metric_vals):
                metric_tot[i] += v

            toc = time.time()
            time_elapsed += (toc - tic)

            fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_vals) for item in tup ])
            log_str = "TEST BATCH-%d: t = %.4f"%(i, time_elapsed) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
            self.logger.info(log_str)
        
        metric_tot = metric_tot/self.config.test.num_batches
        fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_tot) for item in tup ])
        log_str = "TEST ALL-%d: t = %.4f"%(iter, time_elapsed) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
        self.logger.info(log_str)
            
        if self.config.test.save_output:
            np.save(os.path.join(self.config.dir_name, 'alloc_tst_' + str(iter)), alloc_tst)
            np.save(os.path.join(self.config.dir_name, 'pay_tst_' + str(iter)), pay_tst)
            np.save(os.path.join(self.config.dir_name, 'rev_tst_' + str(iter)), rev_tst)
        
        return metric_tot
            
    
