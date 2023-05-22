from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import torch
# import tensorflow as tf


from nets import *
from cfgs import *
from data import *
from clip_ops.clip_ops import *
from trainer import *
from baseline.baseline import *

print(1)
# D =11

# x = np.linspace(0, 1.0, D)
# data = np.stack([v.flatten() for v in np.meshgrid(x,x)], axis = -1)
# X_tst = data[:,0].reshape([data.shape[0], 1])
# Y_tst = data[:,1].reshape([data.shape[0], 1])
# fixed_data = np.full(X_tst.shape, 0.5)
# X_tst = np.concatenate([X_tst, fixed_data], axis=-1)
# Y_tst = np.concatenate([Y_tst, fixed_data], axis=-1)


# print(X_tst)
# print(Y_tst)


# shape = [cfg.train.num_misreports, cfg.train.num_batches * cfg.train.batch_size, cfg.num_buyers]
# np.tile(np.expand_dims(np.linspace(0, 1.0, shape[1]), 0).T, [shape[0], 1, 1, shape[2]*shape[3]]).reshape(shape)
# print(np.tile(np.expand_dims(np.linspace(0, 1.0, shape[0]), 0).T, shape[1]*shape[2]).reshape(shape))


# p = torch.tensor([
#     [
#         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#     ],
#     [
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
#     ],
#     [
#         [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#     ]
# ])
# p = F.softmax(p, dim=-1)
# print(p)
# b_shape = p.shape
# p_val = np.tile(np.linspace(0, 1.0, b_shape[2]), [b_shape[0], b_shape[1], 1])
# p_val = torch.from_numpy(p_val.astype(np.float32))


# print((p_val * p).sum(-1))



data = torch.tensor([
    [0.0, 0.5, 0.2, 0.3, 0.2, 0.7],
    [0.5, 0.4, 0.1, 0.7, 0.6, 0.4],
    [0.8, 0.9, 0.5, 1.0, 1.1, 1.5],
])

b_x_in, s_x_in = torch.split(data, [3, 3], dim=-1)
print(b_x_in.sum(-2))
print(s_x_in)

# Net = paired_net.Net
# cfg.num_buyers = 3
# cfg.num_sellers = 3
# cfg.num_agents = 6
# net = Net(cfg)
# a, p, r = net.inference(data)



# x = torch.tensor([0.1, 0.5, 0.3])
# y = torch.tensor([0.2, 0.2, 0.2])

# z = torch.where(y >= x, 1.0, 0.0) 
# print(z)
# b_x, s_x = torch.split(data, 2, dim=-1)
# alloc = torch.tensor(
#     [
#         [1.0, 0.0],
#         [1.0, 1.0],
#         [0.0, 1.0]
#     ]
# )
# revenue = torch.tensor(
#     [
#         [0.8, 0.6],
#         [0.2, 0.4],
#         [0.5, 1.0]
#     ]
# )
# pay = torch.tensor(
#     [
#         [0.3, 0.3],
#         [1.0, 0.5],
#         [0.6, 0.0]
#     ]
# )

# sorted_b_x, sorted_b_idx = torch.sort(b_x, dim=-1, descending=True)
# sorted_s_x, sorted_s_idx = torch.sort(s_x, dim=-1)

# sorted_buyer_utility = torch.multiply(sorted_b_x, alloc)
# sorted_seller_utility = torch.multiply(sorted_s_x, alloc)

# buyer_utility = torch.gather(sorted_buyer_utility, -1, sorted_b_idx) - pay
# seller_utility = revenue - torch.gather(sorted_seller_utility, -1, sorted_s_idx)

# print(buyer_utility)
# print(seller_utility)


# a = torch.zeros(3, cfg.num_buyers, cfg.num_sellers)
# for i in range(3):
#     for j in range(2):
#         a[i][sorted_x_idx[i][j]][sorted_y_idx[i][j]] = out_data[i][j]
# print(a)



# D = 10
# d = np.linspace(0, 1.0, D)
# data = np.stack([v.flatten() for v in np.meshgrid(d,d,d,d,d,d)], axis = -1)
# X_tst = data[:,:cfg.num_buyers]
# Y_tst = data[:,cfg.num_buyers:]
# cfg.test.num_misreports = 1
# cfg.test.gd_iter = 0
# cfg.test.batch_size = D
# cfg.test.num_batches = int(X_tst.shape[0]/cfg.test.batch_size)
# cfg.test.save_output = True
# print([0.0] * 3)
# X_tst = np.array([[1.0, 2.0, 3.0]])
# Y_tst = np.array([[0.0, 0.0, 0.2]])
# X_tst = torch.from_numpy(X_tst.astype(np.float32))
# w_rgt_init_val = 0.0 if "w_rgt_init_val" not in cfg.train else cfg.train.w_rgt_init_val
# w_rgt = np.array([1.0, 2.0, 3.0])
# w_rgt = torch.from_numpy(w_rgt.astype(np.float32))
# w_rgt.requires_grad_(True)
# print(w_rgt*X_tst)
# print((w_rgt*X_tst).sum())
# w_irp_init_val = 0.0 if "w_irp_init_val" not in cfg.train else cfg.train.w_irp_init_val
# w_irp = np.ones(cfg.num_agents).astype(np.float32) * w_irp_init_val
# w_irp = torch.from_numpy(w_irp.astype(np.float32))
# w_irp.requires_grad_(True)







# d = 2
# eps = 1e-3

# def violate(P_eps, eps):
#   print(np.sum(np.sum(P_eps, 0) > 1 + eps),\
#     np.sum(np.sum(P_eps, 0) < 1 - eps),\
#     np.sum(np.sum(P_eps, 1) > 1 + eps),\
#     np.sum(np.sum(P_eps, 1) < 1 - eps))


# # Sinkhorn-Knoopアルゴリズム
# def sinkhorn_knoop(X, iter=10):
#     # violationの確認
#     print(f'[Iteration = Pre] ', end='')
#     violate(X, eps)

#     # 初期化
#     d = X.shape[0] # Xがテンソルの場合にはどの次元でdoubly stochastic matrixにしたいか注意しよう。
#     r = np.ones((d, 1)) # pytorchの場合はrに関しては最適化したくないので、普通にtorch.onesに置き換えるだけでいいはず

#     xdotr = X.T.dot(r) # Xがテンソルの場合にはどの次元でdoubly stochastic matrixにしたいか注意して内積をとる。
    
#     c = 1 / xdotr
#     xdotc = X.dot(c) # Xがテンソルの場合にはどの次元でdoubly stochastic matrixにしたいか注意して内積をとる。
        
#     r = 1 / xdotc

#     P_eps = np.empty(X.shape)

#     # 繰り返し演算
#     for t in range(iter):
#         c = 1 / X.T.dot(r)
#         r = 1 / X.dot(c)

#         _D1 = np.diag(np.squeeze(r))
#         _D2 = np.diag(np.squeeze(c))
#         P_eps = _D1.dot(X).dot(_D2)
        
#         print(f'[Iteration = {t}] ', end='')
#         violate(P_eps, eps)

#     return P_eps




# X = np.random.randn(d,d+1)
# X = np.abs(X)

# # print(X)
# # print(y)

# # violate(X, eps)

# # x = torch.tensor(
# #     [
# #         [[1.0, 1.0], [1.0, 1.0]],
# #         [[2.0, 2.0], [2.0, 2.0]],
# #         [[3.0, 3.0], [3.0, 3.0]]
# #     ]
# # )
# # data = torch.tensor(
# #     [
# #         [[0.8, 0.6], [0.5, 0.2]],
# #         [[0.2, 0.4], [0.7, 0.9]],
# #         [[0.5, 1.0], [0.3, 0.8]]
# #     ]
# # )



# # # xdotr = (x * data).sum(-1)
# # # print(xdotr)

# # Sinkhorn-Knoopアルゴリズム
# def sinkhorn_knoop(X, iter=100):
#     # violationの確認
#     #   print(f'[Iteration = Pre] ', end='')
#     #   violate(X, eps)

#     # 初期化
#     d = X.shape[-2] # Xがテンソルの場合にはどの次元でdoubly stochastic matrixにしたいか注意しよう。
#     r = torch.ones((X.shape[0], d, 1)) # pytorchの場合はrに関しては最適化したくないので、普通にtorch.onesに置き換えるだけでいいはず
#     xdotr = torch.bmm(X.transpose(-1, -2), r) # Xがテンソルの場合にはどの次元でdoubly stochastic matrixにしたいか注意して内積をとる。
    
#     c = 1 / xdotr
#     xdotc = torch.bmm(X, c) # Xがテンソルの場合にはどの次元でdoubly stochastic matrixにしたいか注意して内積をとる。
        
#     r = 1 / xdotc

#     P_eps = torch.empty(X.shape)
#     # 繰り返し演算
#     for t in range(iter):
#         c = 1 / torch.bmm(X.transpose(-1, -2), r)
#         r = 1 / torch.bmm(X, c)

#         _D1 = torch.diag_embed(torch.squeeze(r))
#         _D2 = torch.diag_embed(torch.squeeze(c))
#         P_eps = torch.bmm(torch.bmm(_D1, X), _D2)

#     return P_eps



# # X = torch.rand(3, 2, 3)
# P = sinkhorn_knoop(X)
# print(P)




