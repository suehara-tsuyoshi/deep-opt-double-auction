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

elif setting == "maxwelf_5x5_uniform":
    cfg = maxwelf_5x5_uniform_config.cfg
    Net = irp_constrained_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_max_welf.Trainer

elif setting == "opt_md_3x3_uniform":
    cfg = opt__md_3x3_uniform_config.cfg
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

else:
    print("None selected")
    sys.exit(0)

save_plot = True
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 'x-large'})
D = 201

x = np.linspace(0, 1.0, D)
data = np.stack([v.flatten() for v in np.meshgrid(x,x)], axis = -1)
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

for i in range(int(max_iter/train_iter)):
    cfg.test.restore_iter += train_iter
    net = Net(cfg)
    generator = Generator(cfg, 'test', X=X_tst, Y=Y_tst)
    m = Trainer(cfg, "test", net, clip_op_lambda)
    metrics = m.test(generator)
    welfare.append(metrics[1])
    buyer_regret.append(metrics[2])
    seller_regret.append(metrics[3])
    regret.append(metrics[4])
    auctioneer_irp.append(metrics[7])
    entropy.append(metrics[8])

plt.plot(np.arange(0, max_iter, train_iter), welfare, label='welfare')
plt.title(setting)
plt.legend()
plt.savefig(os.path.join(dir_name, 'welfare'+'.pdf'))

plt.figure()

plt.plot(np.arange(0, max_iter, train_iter), regret, label='regret')
plt.title(setting)
plt.legend()
plt.savefig(os.path.join(dir_name, 'regret'+'.pdf'))