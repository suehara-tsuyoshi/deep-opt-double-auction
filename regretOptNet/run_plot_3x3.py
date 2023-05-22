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

train_iter = 50000
max_iter = 400000

welfare = np.load(cfg.dir_name + "/welfare400000.npy")
regret = np.load(cfg.dir_name + "/regret400000.npy")
bbp = np.load(cfg.dir_name + "/bbp400000.npy")
entropy = np.load(cfg.dir_name + "/entropy400000.npy")
# welfare = np.load(cfg.dir_name + "/welfare.npy")
# regret = np.load(cfg.dir_name + "/regret.npy")
# bbp = np.load(cfg.dir_name + "/bbp.npy")
# entropy = np.load(cfg.dir_name + "/entropy.npy")

welf = np.load("experiments/maxwelf_3x3_uniform/welfare400000.npy")
reg = np.load("experiments/maxwelf_3x3_uniform/regret400000.npy")
b = np.load("experiments/maxwelf_3x3_uniform/bbp400000.npy")
e = np.load("experiments/maxwelf_3x3_uniform/entropy400000.npy")




plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000),  welfare[:int((max_iter+train_iter)/train_iter)],marker='.', label='RegretNet_us')
plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000),  welf[:int((max_iter+train_iter)/train_iter)],marker='.', label='RegretNet')
plt.plot([0, 80], [0.702556, 0.702556], label='VCG')
plt.plot([0, 80], [0.640415, 0.640415], label='MDP')
plt.ylim([0.4, 0.75])
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Test Welfare', fontsize=12)

# plt.title(setting)

plt.legend()
plt.savefig(os.path.join(cfg.dir_name, 'welfare'+'.png'))

plt.figure()

plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000), regret[:int((max_iter+train_iter)/train_iter)], marker='.',label='RegretNet_us')
plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000), reg[:int((max_iter+train_iter)/train_iter)], marker='.',label='RegretNet')
plt.ylim([0.0, 0.05])
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Test Regret', fontsize=12)
# plt.title(setting)

plt.legend()
plt.savefig(os.path.join(cfg.dir_name, 'regret'+'.png'))

plt.figure()

plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000), bbp[:int((max_iter+train_iter)/train_iter)],marker='.', label='RegretNet_us')
plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000), b[:int((max_iter+train_iter)/train_iter)],marker='.', label='RegretNet')
plt.ylim([-0.02, 0.3])
plt.plot([0, 80], [0.235770, 0.235770], label='VCG', color='orange')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Test BBP', fontsize=12)
# plt.title(setting)

plt.legend()
plt.savefig(os.path.join(cfg.dir_name, 'bbp'+'.png'))

plt.figure()

plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000), entropy[:int((max_iter+train_iter)/train_iter)], marker='.',label='RegretNet_us')
plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000), e[:int((max_iter+train_iter)/train_iter)], marker='.',label='RegretNet')
plt.ylim([0.0, 0.2])
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Test Entropy', fontsize=12)
# plt.title(setting)

plt.legend()
plt.savefig(os.path.join(cfg.dir_name, 'entropy'+'.png'))