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

elif setting == "opt_3x3_uniform":
    cfg = opt_3x3_uniform_config.cfg
    Net = mdp_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer_opt.Trainer

else:
    print("None selected")
    sys.exit(0)


# D = 11
# d = np.linspace(0, 1.0, D)
# data = np.stack([v.flatten() for v in np.meshgrid(d,d,d,d)], axis = -1)
# X_tst = data[:,:cfg.num_buyers]
# Y_tst = data[:,cfg.num_buyers:]
# cfg.test.num_misreports = 1
# cfg.test.gd_iter = 25
# cfg.test.batch_size = D
# cfg.test.num_batches = int(X_tst.shape[0]/cfg.test.batch_size)
# cfg.test.save_output = True
# cfg.test.restore_iter = 200000


D = 1
X_tst = np.array([[1.0]])
Y_tst = np.array([[0.6]])
cfg.test.num_misreports = 1
cfg.test.gd_iter = 25
cfg.test.batch_size = D
cfg.test.num_batches = int(X_tst.shape[0]/cfg.test.batch_size)
cfg.test.save_output = True
cfg.test.restore_iter = 20000

net = Net(cfg)
generator = Generator(cfg, 'test', X = X_tst, Y = Y_tst)
clip_op_lambda = (lambda x: clip_op_01(x))
m = Trainer(cfg, "test", net, clip_op_lambda)
m.test(generator)




