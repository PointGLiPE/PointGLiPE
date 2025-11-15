import os
import h5py
from torch.utils.data import Dataset
import torch
from config import data_path
# 数据加载有两种方式，一是在init中就把所有的h5py转换为tensor文件，这样的好处就速度快，但是占用显存大，不过modelnet40本省就小，无所谓
# 另外一种就是每次用到h5py文件的时候，将用到的转为tensor，这样的好处就是占用显存小，但是慢了，所以一般不采用
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # 保证数据完整性，防止多个进程同时修改导致文件损坏

class ModelNet40(Dataset):
    def __init__(self, num_points=1024, partition='train'):
        paths = data_path
        paths = paths.glob(f"ply_data_{partition}*.h5")
        data, label = [], []
        for p in paths:
            f = h5py.File(p, 'r')
            data.append(torch.from_numpy(f['data'][:]).float())
            label.append(torch.from_numpy(f['label'][:]).long())
            f.close()
        self.data = torch.cat(data)
        self.label = torch.cat(label).squeeze()
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, idx):
        pc = self.data[idx][:self.num_points]  # 这里面限定了1024个点

        label = self.label[idx]
        if self.partition == 'train':
            scale = torch.rand((3,)) * (3/2 - 2/3) + 2/3
            pc = pc * scale
            pc = pc[torch.randperm(pc.shape[0])]

        return pc*40, label

    def __len__(self):
        return self.data.shape[0]
