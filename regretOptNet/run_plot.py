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

train_iter = 50000
max_iter = 400000

welfare_0 = np.load("experiments/maxwelf_3x3_uniform/welfare400000.npy")
regret_0 = np.load("experiments/maxwelf_3x3_uniform/regret400000.npy")
bbp_0 = np.load("experiments/maxwelf_3x3_uniform/bbp400000.npy")
entropy_0 = np.load("experiments/maxwelf_3x3_uniform/entropy400000.npy")

welfare_1 = np.load("experiments/maxwelf_3x3_uniform_1/welfare400000.npy")
regret_1 = np.load("experiments/maxwelf_3x3_uniform_1/regret400000.npy")
bbp_1 = np.load("experiments/maxwelf_3x3_uniform_1/bbp400000.npy")
entropy_1 = np.load("experiments/maxwelf_3x3_uniform_1/entropy400000.npy")

welfare_3 = np.load("experiments/maxwelf_3x3_uniform_3/welfare400000.npy")
regret_3 = np.load("experiments/maxwelf_3x3_uniform_3/regret400000.npy")
bbp_3 = np.load("experiments/maxwelf_3x3_uniform_3/bbp400000.npy")
entropy_3 = np.load("experiments/maxwelf_3x3_uniform_3/entropy400000.npy")


plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000),  welfare_0[:int((max_iter+train_iter)/train_iter)],marker='.', label='0')
plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000),  welfare_1[:int((max_iter+train_iter)/train_iter)],marker='.', label='1')
plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000),  welfare_3[:int((max_iter+train_iter)/train_iter)],marker='.', label='3')
plt.plot([0, 80], [0.702556, 0.702556], label='VCG')
plt.plot([0, 80], [0.640415, 0.640415], label='MDP')
plt.ylim([0.4, 0.75])
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Test Welfare', fontsize=12)

# plt.title(setting)

plt.legend()
plt.savefig(os.path.join('welfare'+'.pdf'))

plt.figure()

plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000),  regret_0[:int((max_iter+train_iter)/train_iter)],marker='.', label='0')
plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000),  regret_1[:int((max_iter+train_iter)/train_iter)],marker='.', label='1')
plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000),  regret_3[:int((max_iter+train_iter)/train_iter)],marker='.', label='3')
plt.ylim([0.0, 0.05])
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Test Regret', fontsize=12)
# plt.title(setting)

plt.legend()
plt.savefig(os.path.join('regret'+'.pdf'))

plt.figure()

plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000),  bbp_0[:int((max_iter+train_iter)/train_iter)],marker='.', label='0')
plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000),  bbp_1[:int((max_iter+train_iter)/train_iter)],marker='.', label='1')
plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000),  bbp_3[:int((max_iter+train_iter)/train_iter)],marker='.', label='3')
plt.ylim([0.0, 0.01])
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Test BBP', fontsize=12)
# plt.title(setting)

plt.legend()
plt.savefig(os.path.join('bbp'+'.pdf'))

plt.figure()

plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000),  entropy_0[:int((max_iter+train_iter)/train_iter)],marker='.', label='0')
plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000),  entropy_1[:int((max_iter+train_iter)/train_iter)],marker='.', label='1')
plt.plot(np.arange(0, (max_iter+train_iter)/5000, train_iter/5000),  entropy_3[:int((max_iter+train_iter)/train_iter)],marker='.', label='3')
plt.ylim([0.0, 0.2])
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Test Entropy', fontsize=12)
# plt.title(setting)

plt.legend()
plt.savefig(os.path.join('entropy'+'.pdf'))