import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn.init import trunc_normal_
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.models.layers import DropPath



def checkpoint(function, *args, **kwargs):  # 正向传播时不保存所有激活值(激活函数的值)，到了反向传播再重新计算这些激活
    return torch_checkpoint(function, *args, use_reentrant=False, **kwargs)


class NCA(nn.Module):
    def __init__(self, dim, out_dim):
        # dim=64
        # out_dim对应的维度
        super().__init__()

        self.mlp1 = nn.Linear(3, dim // 2)
        self.mlp2 = nn.Linear(dim // 2, dim // 2)
        self.mlp3 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, out_dim)
        )

    def forward(self, xyz, knn):
        # x: (N,C)
        N, K = knn.shape
        p_feat0 = self.mlp1(xyz)  # (N 3)->(N C)
        ps0 = p_feat0.index_select(0, knn.view(-1)).view(N, K, -1) - p_feat0.unsqueeze(1)  # (N K C)
        p_local = ps0.max(dim=1).values  # (N C)
        p_feat1 = self.mlp2(p_feat0)  # (N C)
        ps1 = p_feat1.index_select(0, knn.view(-1)).view(N, K, -1) - p_feat1.unsqueeze(1)  # (N K C)
        p_local_exp = p_local.unsqueeze(1).expand_as(ps1)  # (N K C)
        p_feat2 = torch.cat([ps1, p_local_exp], dim=-1)  # (N K 2C)
        mlp_logits = self.mlp3(p_feat2)  # (N K C)
        return mlp_logits


class LFA(nn.Module):
    def __init__(self, dim, out_dim, bn_momentum, init=0.):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.bn = nn.BatchNorm1d(dim, momentum=bn_momentum)
        nn.init.constant_(self.bn.weight, init)
        self.nca = NCA(dim, out_dim)

    def forward(self, xyz, x, knn):
        x = self.proj(x)
        N, K = knn.shape
        xs = x.index_select(0, knn.view(-1)).view(N, K, -1) - x.unsqueeze(1)  # (N K C)
        xyz_pe = self.nca(xyz, knn)  # 位置编码
        xs = xs + xyz_pe  # 原来的每个邻居都加上位置编码
        x = xs.max(dim=1)[0]  # (N C)
        x = self.bn(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_dim, mlp_ratio, bn_momentum, act, init=0.):
        super().__init__()
        hid_dim = round(in_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            act(),
            nn.Linear(hid_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim, momentum=bn_momentum),
        )
        nn.init.constant_(self.mlp[-1].weight, init)

    def forward(self, x):
        x = self.mlp(x)
        return x


class MultiScaleAttentionPE(nn.Module):
    def __init__(self, embed_dim, num_heads=2):
        super().__init__()
        self.mlp_all = nn.Linear(3, embed_dim)

        # 为五个层级分别准备残差/距离映射 mlp
        self.mlp4 = nn.Linear(3, embed_dim)  # top (最小集合)
        self.mlp3 = nn.Linear(3, embed_dim)
        self.mlp2 = nn.Linear(3, embed_dim)
        self.mlp1 = nn.Linear(3, embed_dim)
        self.mlp0 = nn.Linear(3, embed_dim)

        # 投影回 embed_dim
        self.proj4 = nn.Linear(2 * embed_dim, embed_dim)
        self.proj3 = nn.Linear(2 * embed_dim, embed_dim)
        self.proj2 = nn.Linear(2 * embed_dim, embed_dim)
        self.proj1 = nn.Linear(2 * embed_dim, embed_dim)
        self.proj0 = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, xyz0, indices_nn, indices, pts_list):
        """
            pe_list = [pe4, pe3, pe2, pe1, pe0]
        """
        device = xyz0.device

        # ------------- 提取各层坐标（保持你原来的索引布局） -------------
        # 这里沿用原始下标，请确保 indices 的布局与你原来一致
        xyz4 = xyz0[indices[5]]  # 最小集合
        xyz3 = xyz0[indices[7]]
        xyz2 = xyz0[indices[9]]
        xyz1 = xyz0[indices[11]]
        # xyz0 是输入的最大集合，不需要索引

        pe_list = []
        N0 = xyz0.shape[0]
        N1 = xyz1.shape[0]
        N2 = xyz2.shape[0]
        N3 = xyz3.shape[0]
        N4 = xyz4.shape[0]

        # 基础特征一次 mlp_all 然后 slice（与原语义一致）
        f0 = self.mlp_all(xyz0)  # (N0, C)
        f1 = f0[:N1, :]
        f2 = f0[:N2, :]
        f3 = f0[:N3, :]
        f4 = f0[:N4, :]
        _, C = f4.shape

        knn01 = indices_nn[3]
        knn12 = indices_nn[2]
        knn23 = indices_nn[1]
        knn34 = indices_nn[0]


        pe_list.append(f4)
        feat4_pe = f4
        # ---------------- Step: level3 相对于 level4 的 delta 编码 ----------------

        # knn34 形状通常 (N3,1) 或 (N3,)：统一变为一维索引
        knn34_idx = knn34.view(-1).long()

        # 使用 index_select（通常比 fancy indexing 产生更少临时）
        f34 = feat4_pe.index_select(0, knn34_idx).view(N3, -1)  # (N3, C)
        xyz34 = xyz4.index_select(0, knn34_idx).view(N3, 3)  # (N3, 3)
        dist34 = xyz3 - xyz34  # (N3, 3)
        f34_dist = self.mlp3(dist34)  # (N3, C)
        f3_nei = f34 + f34_dist  # (N3, C)
        feat3_pe = torch.cat([f3_nei, f3], dim=-1)  # (N3, 2C)
        feat3_pe = self.proj3(feat3_pe)  # (N3, C)
        pe_list.append(feat3_pe)


        # ---------------- Step: level2 相对于 level3 的 delta 编码 ----------------
        knn23_idx = knn23.view(-1).long()
        f23 = feat3_pe.index_select(0, knn23_idx).view(N2, -1)
        xyz23 = xyz3.index_select(0, knn23_idx).view(N2, 3)
        dist23 = xyz2 - xyz23
        f23_dist = self.mlp2(dist23)
        f2_nei = f23 + f23_dist
        feat2_pe = torch.cat([f2_nei, f2], dim=-1)
        feat2_pe = self.proj2(feat2_pe)
        pe_list.append(feat2_pe)


        # ---------------- Step: level1 相对于 level2 的 delta 编码 ----------------
        knn12_idx = knn12.view(-1).long()
        f12 = feat2_pe.index_select(0, knn12_idx).view(N1, -1)
        xyz12 = xyz2.index_select(0, knn12_idx).view(N1, 3)
        dist12 = xyz1 - xyz12
        f12_dist = self.mlp1(dist12)
        f1_nei = f12 + f12_dist
        feat1_pe = torch.cat([f1_nei, f1], dim=-1)
        feat1_pe = self.proj1(feat1_pe)
        pe_list.append(feat1_pe)

        # ---------------- Step: level0 相对于 level1 的 delta 编码 ----------------
        f01 = feat1_pe.index_select(0, knn01).view(N0, -1)
        xyz01 = xyz1.index_select(0, knn01).view(N0, 3)
        dist01 = xyz0 - xyz01
        f01_dist = self.mlp0(dist01)
        f0_nei = f01 + f01_dist
        feat0_pe = torch.cat([f0_nei, f0], dim=-1)
        feat0_pe = self.proj0(feat0_pe)
        pe_list.append(feat0_pe)

        return pe_list  # [pe4, pe3, pe2, pe1, pe0]


class Block(nn.Module):
    def __init__(self, dim, depth, drop_path, mlp_ratio, bn_momentum, act, heads):
        super().__init__()
        '''
        drop_path:
        [0.0000, 0.0043, 0.0087, 0.0130]
        '''
        self.dim = dim
        self.depth = depth  # 这个不是01234  而是[4 4 4 8 4]中的一个数
        self.lfas = nn.ModuleList([
            LFA(dim, dim, bn_momentum) for _ in range(depth)
        ])
        self.mlp = Mlp(dim, mlp_ratio, bn_momentum, act, 0.2)
        self.mlps = nn.ModuleList([
            Mlp(dim, mlp_ratio, bn_momentum, act) for _ in range(depth // 2)
        ])
        if isinstance(drop_path, list):  # 如果drop_path是个列表，
            drop_rates = drop_path  # 使用这个列表
            self.dp = [dp > 0. for dp in drop_path]  # drop_path>0 则为true  self.dp = [False, True, True, True]
        else:  # drop_path 如果传进来是一个 float，例如 0.1
            drop_rates = torch.linspace(0., drop_path, self.depth).tolist()  # 自动生成线性增大的 drop rate
            self.dp = [drop_path > 0.] * depth  # 如果drop_path>0[True, True, True, True]，否则[False, False, False, False]

        self.drop_paths = nn.ModuleList([
            DropPath(dpr) for dpr in drop_rates
        ])  # 按照[0.0000, 0.0043, 0.0087, 0.0130] 生成DropPath

        self.gpe = nn.Linear(64, dim, bias=False)

    def drop_path(self, x, i, pts):
        if not self.dp[i] or not self.training:  # 如果不是训练  或者  即使是训练但是 self.dp[i]=False 则跳过
            return x
        return torch.cat([self.drop_paths[i](xx) for xx in torch.split(x, pts, dim=0)], dim=0)
        # x 按照 pts分割，然后按照概率把其中的某一部分置零，然后在拼接在一起

    def forward(self, xyz, x, knn_raw, g_pos, pts=None):
        x = x + self.gpe(g_pos)
        x = x + self.drop_path(self.mlp(x), 0, pts)
        for i in range(self.depth):
            if i % 2 == 1:
                x = x + self.drop_path(self.lfas[i](xyz, x, knn_raw), i, pts)
                x = x + self.drop_path(self.mlps[i // 2](x), i, pts)
        return x


class Stage(nn.Module):
    def __init__(self, args, depth=0):
        super().__init__()

        self.depth = depth
        self.up_depth = len(args.depths) - 1

        self.first = first = depth == 0
        self.last = last = depth == self.up_depth

        self.k = args.ks[depth]

        self.cp = cp = args.use_cp
        cp_bn_momentum = args.cp_bn_momentum if cp else args.bn_momentum

        dim = args.dims[depth]
        nbr_in_dim = 10 if first else 3
        nbr_hid_dim = args.nbr_dims[0] if first else args.nbr_dims[1] // 2
        nbr_out_dim = dim if first else args.nbr_dims[1]
        self.nbr_embed = nn.Sequential(
            nn.Linear(nbr_in_dim, nbr_hid_dim // 2, bias=False),
            nn.BatchNorm1d(nbr_hid_dim // 2, momentum=cp_bn_momentum),
            args.act(),
            nn.Linear(nbr_hid_dim // 2, nbr_hid_dim, bias=False),
            nn.BatchNorm1d(nbr_hid_dim, momentum=cp_bn_momentum),
            args.act(),
            nn.Linear(nbr_hid_dim, nbr_out_dim, bias=False),
        )
        self.nbr_bn = nn.BatchNorm1d(dim, momentum=args.bn_momentum)
        nn.init.constant_(self.nbr_bn.weight, 0.8 if first else 0.2)
        self.nbr_proj = nn.Identity() if first else nn.Linear(nbr_out_dim, dim, bias=False)

        if not first:
            in_dim = args.dims[depth - 1]
            self.skip_proj = nn.Sequential(
                nn.Linear(in_dim, dim, bias=False),
                nn.BatchNorm1d(dim, momentum=args.bn_momentum)
            )
            nn.init.constant_(self.skip_proj[1].weight, 0.3)

        self.blk = Block(dim, args.depths[depth],
                         args.drop_paths[depth],
                         args.mlp_ratio, cp_bn_momentum, args.act, args.heads[depth])

        '''args.drop_paths[depth] 按照depth每次取一列
        glipe_args.drop_paths = [
                                [0.0000, 0.0043, 0.0087, 0.0130],
                                [0.0174, 0.0217, 0.0261, 0.0304],
                                ...
                                ]
        '''

        self.drop = DropPath(args.head_drops[depth])  # [0.0, 0.05, 0.1, 0.15, 0.2]  用在了解码器上
        # DropPath(0.2) 以 20% 的概率将整条残差路径（residual connection）置为 0（即 drop 掉），以 80% 的概率保留它
        self.postproj = nn.Sequential(
            nn.BatchNorm1d(dim, momentum=args.bn_momentum),
            nn.Linear(dim, args.head_dim, bias=False),
        )
        nn.init.constant_(self.postproj[0].weight, (args.dims[0] / dim) ** 0.5)

        self.cor_std = 1 / args.cor_std[depth]
        self.cor_head = nn.Sequential(
            nn.Linear(dim, 32, bias=False),
            nn.BatchNorm1d(32, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(32, 3, bias=False),
        )

        if not last:
            self.sub_stage = Stage(args, depth + 1)

    def local_aggregation(self, xyz, x, knn, g_pos, pts):
        x = self.blk(xyz, x, knn, g_pos, pts)
        return x

    def forward(self, x, xyz, prev_knn, indices, pts_list, pe_list):
        """
        x: N x C
        """
        # downsampling
        if not self.first:
            ids = indices.pop()
            xyz = xyz[ids]
            x = x[ids]
            x = self.skip_proj(x)

        knn = indices.pop()

        # spatial encoding
        N, k = knn.shape
        nbr = xyz.index_select(0, knn.view(-1)).view(N, k, -1) - xyz.unsqueeze(1)
        nbr = torch.cat([nbr, x.index_select(0, knn.view(-1)).view(N, k, -1)], dim=-1).view(-1, 10) if self.first else nbr.view(-1, 3)
        if self.training and self.cp:
            nbr.requires_grad_()
        nbr_embed_func = lambda x: self.nbr_embed(x).view(N, k, -1).max(dim=1)[0]
        nbr = checkpoint(nbr_embed_func, nbr) if self.training and self.cp else nbr_embed_func(nbr)
        nbr = self.nbr_proj(nbr)
        nbr = self.nbr_bn(nbr)
        x = nbr if self.first else nbr + x
        g_pos = pe_list.pop()
        # main block
        knn = knn.unsqueeze(0)
        pts = pts_list.pop() if pts_list is not None else None

        if self.training and self.cp:  # self.cp不知道干嘛用的，反正是False
            x = checkpoint(self.local_aggregation, xyz, x, knn.squeeze(0), g_pos, pts)
        else:  # 肯定选择这个
            x = self.local_aggregation(xyz, x, knn.squeeze(0), g_pos, pts)

        # get subsequent feature maps
        if not self.last:
            sub_x, sub_c = self.sub_stage(x, xyz, knn, indices, pts_list, pe_list)
        else:
            sub_x = sub_c = None

        # regularization
        if self.training:
            rel_k = torch.randint(self.k, (N, 1), device=x.device)
            rel_k = torch.gather(knn.squeeze(0), 1, rel_k).squeeze(1)
            rel_cor = (xyz[rel_k] - xyz)
            rel_cor.mul_(self.cor_std)
            # print(rel_cor.std(dim=0))
            rel_p = x[rel_k] - x
            rel_p = self.cor_head(rel_p)
            closs = F.mse_loss(rel_p, rel_cor)
            sub_c = sub_c + closs if sub_c is not None else closs

        # upsampling
        x = self.postproj(x)
        if not self.first:
            back_nn = indices[self.depth - 1]
            x = x[back_nn]
        x = self.drop(x)  # [0.0, 0.05, 0.1, 0.15, 0.2] 按照depth选择，depth越大，置零的可能性越大
        sub_x = sub_x + x if sub_x is not None else x

        return sub_x, sub_c


class GLiPE(nn.Module):
    def __init__(self, args):
        super().__init__()

        # bn momentum for checkpointed layers
        args.cp_bn_momentum = 1 - (1 - args.bn_momentum) ** 0.5

        self.stage = Stage(args)

        hid_dim = args.head_dim
        out_dim = args.num_classes

        self.head = nn.Sequential(
            nn.BatchNorm1d(hid_dim, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(hid_dim, out_dim)
        )

        self.apply(self._init_weights)

        self.pos = MultiScaleAttentionPE(64)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, x, indices, pts_list=None):

        indices_idx = [4, 7, 10, 13]  # 多点在少点中的最近邻，用在位置编码中的
        indices_nn = [indices[i] for i in indices_idx]
        indices_ori = [v for i, v in enumerate(indices) if i not in indices_idx]

        g_pos = self.pos(xyz, indices_nn, indices_ori, pts_list)
        x, closs = self.stage(x, xyz, None, indices_ori, pts_list, g_pos)
        if self.training:
            return self.head(x), closs
        return self.head(x)


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from scannetv2 import ScanNetV2, scan_collate_fn

    from config import scan_args, scan_warmup_args, glipe_args, batch_size, learning_rate as lr, epoch, warmup, \
        label_smoothing as ls
    import os
    #
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    #
    # from pathlib import Path
    # import torch
    # from time import time, sleep
    #
    # testdlr = DataLoader(ScanNetV2(scan_args, partition="val", loop=1, train=False), batch_size=1,
    #                      collate_fn=scan_collate_fn, pin_memory=True,
    #                      persistent_workers=True, num_workers=16)
    #
    # now = time()
    # model = GLiPE(glipe_args).cuda()
    # for xyz, feature, indices, pts, y in testdlr:
    #     xyz = xyz.cuda(non_blocking=True)
    #     feature = feature.cuda(non_blocking=True)
    #     indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
    #     pts = pts.tolist()[::-1]
    #     y = y.cuda(non_blocking=True)
    #     model(xyz, feature, indices, pts)
    #     break
    # print('======================测试成功====================')
    # b = time() - now
    # print(b)
    def summarize_model_params(model):
        total_params = 0
        trainable_params = 0
        frozen_params = 0

        print(f"{'Layer Name':40} | {'Param #':15} | {'Trainable':10}")
        print("-" * 70)
        for name, param in model.named_parameters():
            num = param.numel()
            total_params += num
            if param.requires_grad:
                trainable_params += num
                trainable = "Yes"
            else:
                frozen_params += num
                trainable = "No"
            print(f"{name:40} | {num:15,} | {trainable:10}")

        print("-" * 70)
        print(f"{'Total':40} | {total_params:15,} |")
        print(f"{'Trainable':40} | {trainable_params:15,} |")
        print(f"{'Frozen':40} | {frozen_params:15,} |")

        return total_params, trainable_params, frozen_params
    model = GLiPE(glipe_args).cuda()
    # 使用方法
    summarize_model_params(model)
