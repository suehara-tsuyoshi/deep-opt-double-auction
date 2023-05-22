from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
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

np.random.seed(cfg.test.seed)
generator = Generator(cfg, 'test')
data_b = np.array([next(generator.gen_func)[0] for _ in range(cfg.test.num_batches)])
data_s = np.array([next(generator.gen_func)[2] for _ in range(cfg.test.num_batches)])
data_b = np.squeeze(data_b.reshape(-1, cfg.num_buyers))
data_s = np.squeeze(data_s.reshape(-1, cfg.num_sellers))

print("OPT: ")
print(MDProtcol(cfg, data_b, data_s).opt_rev())

