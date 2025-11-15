import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from pathlib import Path
import sys
from typing import Sequence

sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.models.layers import DropPath
from utils.cutils import knn_edge_maxpooling
from pointnet2_ops import pointnet2_utils
from torch.amp import autocast


@autocast(enabled=False, device_type='cuda')
def calc_pwd(x):
    x2 = x.square().sum(dim=2, keepdim=True)
    return x2 + x2.transpose(1, 2) + torch.bmm(x, x.transpose(1, 2).mul(-2))


def index_points(points, idx):
    """
    通用索引函数：根据 idx 从 points 中提取子点。
    Args:
        points: Tensor, shape [B, N, C]
        idx:    LongTensor, shape [B, M, k] 或 [B, M]
    Returns:
        new_points: Tensor, shape [B, M, k, C] 或 [B, M, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    C = points.shape[-1]
    if idx.dim() == 3:
        # (B, M, k) -> (B, M, k, C)
        view_shape.append(1)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, C)
        pts_expanded = points.unsqueeze(1).expand(-1, view_shape[1], -1, -1)
        return torch.gather(pts_expanded, 2, idx_expanded)
    else:
        # (B, M) -> (B, M, C)
        view_shape.append(C)
        batch_indices = torch.arange(B, device=points.device).view(B, *([1] * (idx.dim() - 1))).expand_as(idx)
        return points[batch_indices, idx, :]


def knn(a: torch.Tensor, b: torch.Tensor, K: int) -> torch.Tensor:
    """
    在 b 中为 a 的每个点找最近的 K 个邻居索引。

    Args:
        a: Tensor of shape (B, N, C)
        b: Tensor of shape (B, M, C)
        K: 邻居数量

    Returns:
        idx: LongTensor of shape (B, N, K)，表示 b 中对应的最近 K 个点的索引
    """
    # 计算批内所有点对的欧氏距离矩阵，形状 (B, N, M)
    # torch.cdist 会自动优化，且支持 GPU 并行
    dist = torch.cdist(a, b)

    # 取最小的 K 个距离对应的索引
    idx = dist.topk(K, dim=-1, largest=False).indices
    return idx


def get_graph_feature(x, idx):
    B, N, C = x.shape
    k = idx.shape[-1]
    nbr = torch.gather(x, 1, idx.view(B, N * k, 1).expand(-1, -1, C)).view(B * N, k, C)
    x = x.view(B * N, 1, C).expand(-1, k, -1)
    return nbr - x


def get_nbr_feature(x, idx):
    B, N, k = idx.shape
    C = x.shape[-1]
    nbr = torch.gather(x, 1, idx.view(B, N * k, 1).expand(-1, -1, C)).view(B * N * k, C)
    return nbr


class NCA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mlp1 = nn.Linear(3, dim // 2)
        self.mlp2 = nn.Linear(dim // 2, dim // 2)
        self.mlp3 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, xyz):
        # x: (B,N,K,C)
        p_feat0 = self.mlp1(xyz)  # (B N K 3)->(B N K C)
        p_local = p_feat0.max(dim=2).values  # (B N C)
        p_feat1 = self.mlp2(p_feat0)  # (B N K C)
        p_local_exp = p_local.unsqueeze(2).expand_as(p_feat1)  # (B N K C)
        p_feat2 = torch.cat([p_feat1, p_local_exp], dim=-1)  # (B N K 2C)
        mlp_logits = self.mlp3(p_feat2)  # (B N K C)
        return mlp_logits


class LFA(nn.Module):
    r"""
    Local Feature Propagation Layer
    f = linear(f)
    f_i = bn(max{f_j | j in knn_i} - f_i)
    """

    def __init__(self, in_dim, out_dim, bn_momentum, init=0.):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, momentum=bn_momentum)
        nn.init.constant_(self.bn.weight, init)

        self.nca = NCA(out_dim)

    def forward(self, xyz, x, knn):
        B, N, C = x.shape
        x = self.proj(x)
        xs = index_points(x, knn) - x.unsqueeze(2)  # (B N K C)

        xyzs = index_points(xyz, knn) - xyz.unsqueeze(2)
        xyz_pe = self.nca(xyzs)

        xs = xs + xyz_pe
        xs_max = xs.max(dim=2)[0]  # (B N C)

        xs_max = self.bn(xs_max.view(B * N, -1)).view(B, N, -1)

        return xs_max


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
        B, N, C = x.shape
        x = self.mlp(x.view(B * N, -1)).view(B, N, -1)
        return x


class Block(nn.Module):
    def __init__(self, dim, depth, drop_path, mlp_ratio, bn_momentum, act, heads):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.mlp = Mlp(dim, mlp_ratio, bn_momentum, act, 0.2)
        self.mlps = nn.ModuleList([
            Mlp(dim, mlp_ratio, bn_momentum, act) for _ in range(depth // 2)
        ])
        if isinstance(drop_path, list):
            drop_rates = drop_path
        else:
            drop_rates = torch.linspace(0., drop_path, self.depth).tolist()
        self.drop_paths = nn.ModuleList([
            DropPath(dpr) for dpr in drop_rates
        ])

        self.lfa0 = LFA(dim, dim, bn_momentum)
        self.lfa1 = LFA(dim, dim, bn_momentum)
        self.lfa2 = LFA(dim, dim, bn_momentum)
        self.lfa3 = LFA(dim, dim, bn_momentum)

        self.gpe = nn.Linear(64, dim, bias=False)

    def forward(self, xyz, x, knn_raw, g_pos):

        x = x + self.gpe(g_pos)

        x = x + self.drop_paths[0](self.mlp(x))

        x = x + self.drop_paths[0](self.lfa0(xyz,x, knn_raw))
        x = x + self.drop_paths[1](self.lfa1(xyz,x, knn_raw))
        x = x + self.drop_paths[1](self.mlps[0](x))

        x = x + self.drop_paths[2](self.lfa2(xyz,x, knn_raw))
        x = x + self.drop_paths[3](self.lfa3(xyz,x, knn_raw))
        x = x + self.drop_paths[3](self.mlps[1](x))

        return x


class Stage(nn.Module):
    def __init__(self, args, depth=0):
        super().__init__()

        self.depth = depth

        self.first = first = depth == 0
        self.last = last = depth == len(args.depths) - 1

        self.n = args.ns[depth]
        self.k = args.ks[depth]

        dim = args.dims[depth]
        nbr_in_dim = 4 if self.first else 3
        nbr_hid_dim = args.nbr_dims[0] if first else args.nbr_dims[1] // 2
        nbr_out_dim = dim if first else args.nbr_dims[1]
        self.nbr_embed = nn.Sequential(
            nn.Linear(nbr_in_dim, nbr_hid_dim // 2, bias=False),
            nn.BatchNorm1d(nbr_hid_dim // 2, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(nbr_hid_dim // 2, nbr_hid_dim, bias=False),
            nn.BatchNorm1d(nbr_hid_dim, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(nbr_hid_dim, nbr_out_dim, bias=False)
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

        self.blk = Block(dim, args.depths[depth], args.drop_paths[depth], args.mlp_ratio, args.bn_momentum, args.act,
                         args.heads[depth])

        if not last:
            self.sub_stage = Stage(args, depth + 1)

    def forward(self, x, xyz, prev_knn, pwd, pe_list):
        """
        x: B x N x C
        """
        # downsampling
        if not self.first:
            xyz = xyz[:, :self.n].contiguous()
            B, N, C = x.shape
            x = self.skip_proj(x.view(B * N, C)).view(B, N, -1)[:, :self.n]

        _, knn = pwd[:, :self.n, :self.n].topk(k=self.k, dim=-1, largest=False, sorted=False)

        # spatial encoding
        B, N, k = knn.shape
        nbr = get_graph_feature(xyz, knn).view(-1, 3)
        if self.first:
            height = xyz[..., 1:2] / 10
            height -= height.min(dim=1, keepdim=True)[0]
            if self.training:
                height += torch.empty((B, 1, 1), device=xyz.device).uniform_(-0.2, 0.2) * 4
            nbr = torch.cat([nbr, get_nbr_feature(height, knn)], dim=1)

        nbr = self.nbr_embed(nbr).view(B * N, k, -1).max(dim=1)[0]
        nbr = self.nbr_proj(nbr)
        nbr = self.nbr_bn(nbr).view(B, N, -1)
        x = nbr if self.first else nbr + x  # 取消位置编码 首次有，以后都没了  x.contiguous()
        g_pe = pe_list.pop()
        # main block
        x = self.blk(xyz, x, knn, g_pe)

        # next stage
        if not self.last:
            sub_x = self.sub_stage(x, xyz, knn, pwd, pe_list)
        else:
            sub_x = x

        return sub_x


class MultiScaleAttentionPE(nn.Module):
    def __init__(self, embed_dim, num_heads=2):
        super().__init__()

        self.mlp_all = nn.Linear(3, embed_dim)

        # 残差映射
        self.mlp2 = nn.Linear(3, embed_dim)
        self.mlp1 = nn.Linear(3, embed_dim)
        self.mlp0 = nn.Linear(3, embed_dim)
        self.proj2 = nn.Linear(2 * embed_dim, embed_dim)
        self.proj1 = nn.Linear(2 * embed_dim, embed_dim)
        self.proj0 = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, xyz0, xyz1, xyz2, pwd):
        """
        xyz0: (B,1024,3)
        xyz1: (B,256, 3)
        xyz2: (B, 64,  3)
        returns:
          pe2: (B, 64, C)   # step1 得到的 64 点位置编码
          pe1: (B,256, C)   # step2 在 256 点上的位置编码
          pe0: (B,1024,C)   # step3 在 1024 点上的位置编码
        """
        pe_list = []
        B, N0, _ = xyz0.shape
        _, N1, _ = xyz1.shape
        _, N2, _ = xyz2.shape
        f0 = self.mlp_all(xyz0)  # (B 1024 C)
        f1 = f0[:, :N1, :]  # (B 256 C)
        f2 = f0[:, :N2, :]  # (B 64 C)

        # ——— Step1: 对 64 点做 MLP -> (B,64,C)
        cls2 = f2.max(dim=1, keepdim=True)[0]  # (B 1 C)
        cls2 = cls2.expand(-1, N2, -1)  # (B N C)
        f2all_dist = self.mlp2(xyz2)
        f2_nei = cls2 + f2all_dist
        feat2_pe = torch.cat([f2_nei, f2], dim=-1)  # (B N 2C)
        feat2_pe = self.proj2(feat2_pe)  # (B 64 128)->(B,64,64)
        pe_list.append(feat2_pe)
        # ——— Step2: 256 点相对于 64 点的 delta 编码 ———
        # 找到每个 xyz1 对应最近的 xyz2
        _, knn12 = pwd[:, :N1, :N2].topk(k=1, dim=-1, largest=False)  # (B 256 1)
        f12 = index_points(feat2_pe, knn12).squeeze(2)  # (B N C) 这是分配的点特征
        xyz12 = index_points(xyz2, knn12).squeeze(2)  # (B N C)这是分配的点坐标
        dist12 = xyz1 - xyz12
        f12_dist = self.mlp1(dist12)
        f1_nei = f12 + f12_dist
        feat1_pe = torch.cat([f1_nei, f1], dim=-1)
        feat1_pe = self.proj1(feat1_pe)  # (B,256,C)
        pe_list.append(feat1_pe)

        # ——— Step3: 1024 点相对于 256 点的 delta 编码 ———
        _, knn01 = pwd[:, :N0, :N1].topk(k=1, dim=-1, largest=False)
        f01 = index_points(feat1_pe, knn01).squeeze(2)
        xyz01 = index_points(xyz1, knn01).squeeze(2)

        dist01 = xyz0 - xyz01
        f01_dist = self.mlp0(dist01)
        f0_nei = f01 + f01_dist
        feat0_pe = torch.cat([f0_nei, f0], dim=-1)
        feat0_pe = self.proj0(feat0_pe)  # (B,256,C)
        pe_list.append(feat0_pe)

        return pe_list


class GLiPE(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.stage = Stage(args)

        in_dim = args.dims[-1]
        bottleneck = args.bottleneck

        self.proj = nn.Sequential(
            nn.BatchNorm1d(in_dim, momentum=args.bn_momentum),
            nn.Linear(in_dim, bottleneck),
            args.act()
        )

        in_dim = bottleneck
        out_dim = args.num_classes

        self.head = nn.Sequential(
            nn.Linear(bottleneck, 512, bias=False),
            nn.BatchNorm1d(512, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256, momentum=args.bn_momentum),
            args.act(),
            nn.Dropout(.5),
            nn.Linear(256, out_dim)
        )

        self.apply(self._init_weights)
        self.pe = MultiScaleAttentionPE(64)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz):
        if not self.training:
            idx = pointnet2_utils.furthest_point_sample(xyz, 1024).long()
            xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        pwd = calc_pwd(xyz)

        xyz0 = xyz[:, :1024].contiguous()
        xyz1 = xyz[:, :256].contiguous()
        xyz2 = xyz[:, :64].contiguous()
        pe_list = self.pe(xyz0, xyz1, xyz2, pwd)

        x = self.stage(None, xyz, None, pwd, pe_list)
        B, N, _ = x.shape
        x = self.proj(x.view(B * N, -1)).view(B, N, -1).max(dim=1)[0]

        return self.head(x)


if __name__ == '__main__':
    from config import glipe_args

    model = GLiPE(glipe_args).cuda()
    # (32 2048 3) (32 2048 3) (32)->(65536 50)
    xyz = torch.rand(32, 1024, 3).cuda()
    x = model(xyz)

    # 参数计算
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


    print('======================测试成功====================')
