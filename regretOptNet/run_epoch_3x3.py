from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from calendar import c

import os
import sys
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt

from nets import *
from cfgs import *
from data import *
from clip_ops.clip_ops import *
from trainer import *

print("Setting: %s"%(sys.argv[1]))
setting = sys.argv[1]

if setting == "maxwelf_1x1_uniform":
    cfg = maxwelf_1x1_uniform_config.cfg
    Net = irp_constrained_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_max_welf.Trainer

elif setting == "maxwelf_1x2_uniform":
    cfg = maxwelf_1x2_uniform_config.cfg
    Net = irp_constrained_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_max_welf.Trainer

elif setting == "maxwelf_2x1_uniform":
    cfg = maxwelf_2x1_uniform_config.cfg
    Net = irp_constrained_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_max_welf.Trainer

elif setting == "maxwelf_2x2_uniform":
    cfg = maxwelf_2x2_uniform_config.cfg
    Net = irp_constrained_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_max_welf.Trainer

elif setting == "maxwelf_3x3_uniform":
    cfg = maxwelf_3x3_uniform_config.cfg
    Net = irp_constrained_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_max_welf.Trainer

elif setting == "maxwelf_3x3_uniform_1":
    cfg = maxwelf_3x3_uniform_1_config.cfg
    Net = irp_constrained_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_max_welf.Trainer

elif setting == "maxwelf_3x3_uniform_3":
    cfg = maxwelf_3x3_uniform_3_config.cfg
    Net = irp_constrained_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_max_welf.Trainer

elif setting == "maxwelf_3x3_uniform_5":
    cfg = maxwelf_3x3_uniform_5_config.cfg
    Net = irp_constrained_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_max_welf.Trainer

elif setting == "maxwelf_3x3_uniform_unsorted":
    cfg = maxwelf_3x3_uniform_unsorted_config.cfg
    Net = unsorted_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_max_welf.Trainer

elif setting == "maxwelf_3x3_uniform_10":
    cfg = maxwelf_3x3_uniform_10_config.cfg
    Net = irp_constrained_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_max_welf.Trainer

elif setting == "maxwelf_5x5_uniform":
    cfg = maxwelf_5x5_uniform_config.cfg
    Net = irp_constrained_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_max_welf.Trainer

elif setting == "opt_md_2x2_uniform":
    cfg = opt_md_2x2_uniform_config.cfg
    Net = mdp_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_opt.Trainer

elif setting == "opt_vcg_2x2_uniform":
    cfg = opt_vcg_3x3_uniform_config.cfg
    Net = vcg_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_opt.Trainer

elif setting == "opt_md_2x2_uniform":
    cfg = opt_md_3x3_uniform_config.cfg
    Net = mdp_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_opt.Trainer

elif setting == "opt_vcg_3x3_uniform":
    cfg = opt_vcg_3x3_uniform_config.cfg
    Net = vcg_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_opt.Trainer

elif setting == "opt_md_5x5_uniform":
    cfg = opt_md_5x5_uniform_config.cfg
    Net = mdp_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_opt.Trainer

elif setting == "opt_vcg_5x5_uniform":
    cfg = opt_vcg_5x5_uniform_config.cfg
    Net = vcg_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_opt.Trainer

else:
    print("None selected")
    sys.exit(0)


save_plot = True
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 'x-large'})
D = 11

x = np.linspace(0, 1.0, D)
data = np.stack([v.flatten() for v in np.meshgrid(x,x,x,x,x,x)], axis = -1)
X_tst = data[:,:cfg.num_buyers]
Y_tst = data[:,cfg.num_buyers:]

cfg.test.num_misreports = 1
cfg.test.gd_iter = 0
cfg.test.batch_size = D
cfg.test.num_batches = int(X_tst.shape[0]/cfg.test.batch_size)
cfg.test.save_output = True



train_iter = 50000
max_iter = 400000
dir_name = os.path.join("experiments", setting)
cfg.test.restore_iter = 0

buyer_regret = []
seller_regret = []
regret = []
welfare = []
auctioneer_irp = []
entropy = []

vcg_welf = []
md_welf = []

net = Net(cfg)
m = Trainer(cfg, "test", net, clip_op_lambda)

for i in range(int(max_iter/train_iter)+1):
    
    generator = Generator(cfg, 'test', X=X_tst, Y=Y_tst)
    metrics = m.test(generator)
    welfare.append(metrics[1])
    buyer_regret.append(metrics[2])
    seller_regret.append(metrics[3])
    regret.append(metrics[4])
    auctioneer_irp.append(metrics[7])
    entropy.append(metrics[8])
  
    cfg.test.restore_iter += train_iter

np.save(os.path.join(cfg.dir_name, 'welfare'+str(max_iter)), welfare)
np.save(os.path.join(cfg.dir_name, 'regret'+str(max_iter)), regret)
np.save(os.path.join(cfg.dir_name, 'buyer_regret'+str(max_iter)), buyer_regret)
np.save(os.path.join(cfg.dir_name, 'seller_regret'+str(max_iter)), seller_regret)
np.save(os.path.join(cfg.dir_name, 'bbp'+str(max_iter)), auctioneer_irp)
np.save(os.path.join(cfg.dir_name, 'entropy'+str(max_iter)), entropy)
# np.save(os.path.join(cfg.dir_name, 'welfare'), welfare)
# np.save(os.path.join(cfg.dir_name, 'regret'), regret)
# np.save(os.path.join(cfg.dir_name, 'buyer_regret'), buyer_regret)
# np.save(os.path.join(cfg.dir_name, 'seller_regret'), seller_regret)
# np.save(os.path.join(cfg.dir_name, 'bbp'), auctioneer_irp)
# np.save(os.path.join(cfg.dir_name, 'entropy'), entropy)

# plt.plot(np.arange(0, max_iter+train_iter, train_iter), welfare, label='welfare')
# plt.title(setting)
# plt.legend()
# plt.savefig(os.path.join(dir_name, 'welfare'+'.pdf'))

# plt.figure()

# plt.plot(np.arange(0, max_iter+train_iter, train_iter), regret, label='regret')
# plt.title(setting)
# plt.legend()
# plt.savefig(os.path.join(dir_name, 'regret'+'.pdf'))

# plt.figure()

# plt.plot(np.arange(0, max_iter+train_iter, train_iter), auctioneer_irp, label='budget-balanced penalty')
# plt.title(setting)
# plt.legend()
# plt.savefig(os.path.join(dir_name, 'bbp'+'.pdf'))