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
X_tst = data[:,0].reshape([data.shape[0], 1])
Y_tst = data[:,1].reshape([data.shape[0], 1])
fix_val = 0.5
fixed_data = np.full(X_tst.shape, fix_val)
X_tst = np.concatenate([X_tst, fixed_data], axis=-1)
Y_tst = np.concatenate([Y_tst, fixed_data], axis=-1)
# X_tst[:,:, 0] = X_tst[:,:, 0] * 4.0
# X_tst[:,:, 1] = X_tst[:,:, 1] * 3.0




cfg.test.num_misreports = 1
cfg.test.gd_iter = 0
cfg.test.batch_size = D
cfg.test.num_batches = int(X_tst.shape[0]/cfg.test.batch_size)
cfg.test.save_output = True

# adapted parameters
# X_tst[:,:, 0] = X_tst[:,:, 0] * 4.0
# X_tst[:,:, 1] = X_tst[:,:, 1] * 3.0
cfg.test.restore_iter = 400000
iter = cfg.test.restore_iter
ext = [0,1,0,1]
asp = 1.0

net = Net(cfg)
generator = Generator(cfg, 'test', X=X_tst, Y=Y_tst)
m = Trainer(cfg, "test", net, clip_op_lambda)
m.test(generator)

# alloc = np.load(cfg.dir_name + "/alloc_tst_400000.npy").reshape(D,D,2)
# pay = np.load(cfg.dir_name + "/pay_tst_400000.npy").reshape(D,D,1)

alloc = np.load(cfg.dir_name + "/alloc_tst_" + str(iter) + ".npy").reshape(D,D,4)
pay = np.load(cfg.dir_name + "/pay_tst_" + str(iter) + ".npy").reshape(D,D,2)

# alloc_1 = alloc.sum(-1)
# alloc_2 = alloc.sum(-2)

# alloc_1.reshape(D,D,1)
# alloc_2.reshape(D,D,1)


plt.rcParams.update({'font.size': 10, 'axes.labelsize': 'x-large'})
fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize=(8,6))

img = ax.imshow(pay[::-1, :, 0], extent=ext, vmin = 0.0, vmax=1.0, cmap = 'YlOrRd', aspect=asp)
plt.title('Prob. of payment buyer 1')
_ = plt.colorbar(img, fraction=0.046, pad=0.04)

if save_plot:
    fig.set_size_inches(4, 3)
    plt.savefig(os.path.join(cfg.dir_name, "pay" + str(iter) + "_1_fix_" + str(fix_val) +  ".pdf"), bbox_inches = 'tight', pad_inches = 0.05)

plt.figure()

plt.rcParams.update({'font.size': 10, 'axes.labelsize': 'x-large'})
fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize=(8,6))

img = ax.imshow(alloc[::-1, :, 0], extent=ext, vmin = 0.0, vmax=1.0, cmap = 'YlOrRd', aspect=asp)
plt.title('Prob. of allocation')
_ = plt.colorbar(img, fraction=0.046, pad=0.04)

if save_plot:
    fig.set_size_inches(4, 3)
    plt.savefig(os.path.join(cfg.dir_name, "alloc" + str(iter) + "_1_fix_" + str(fix_val) + ".pdf"), bbox_inches = 'tight', pad_inches = 0.05)

plt.figure()

plt.rcParams.update({'font.size': 10, 'axes.labelsize': 'x-large'})
fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize=(8,6))

# img = ax.imshow(alloc[::-1, :, 1], extent=ext, vmin = 0.0, vmax=1.0, cmap = 'YlOrRd', aspect=asp)
# plt.title('Prob. of allocation item 2')
# _ = plt.colorbar(img, fraction=0.046, pad=0.04)

# if save_plot:
#     fig.set_size_inches(4, 3)
#     plt.savefig(os.path.join(cfg.dir_name, "alloc" + str(iter) + "_2.pdf"), bbox_inches = 'tight', pad_inches = 0.05)


