from types import SimpleNamespace
from torch import nn
import torch
from pathlib import Path
data_path = Path("/home/yy/data/modelnet40_ply_hdf5_2048")

epoch = 600
warmup = 60
batch_size = 32
learning_rate = 2e-3
label_smoothing = 0.2

glipe_args = SimpleNamespace()
glipe_args.heads = [2, 4, 8]
glipe_args.depths = [4, 4, 4]
glipe_args.ns = [1024, 256, 64]
glipe_args.ks = [20, 20, 20]
glipe_args.dims = [96, 192, 384]


glipe_args.nbr_dims = [48, 48]
glipe_args.bottleneck = 2048
glipe_args.num_classes = 40
drop_path = 0.15
drop_rates = torch.linspace(0., drop_path, sum(glipe_args.depths)).split(glipe_args.depths)
glipe_args.drop_paths = [dpr.tolist() for dpr in drop_rates]
'''
[
 [0.0000, 0.0136, 0.0273, 0.041],
 [0.0545, 0.0682, 0.0818, 0.0955],
 [0.1091, 0.1227, 0.1364, 0.1500]
]
'''
glipe_args.bn_momentum = 0.1
glipe_args.act = nn.GELU
glipe_args.mlp_ratio = 2
glipe_args.cor_std = [2.8, 5.3, 10]