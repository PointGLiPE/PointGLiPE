import argparse
import importlib
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
from pathlib import Path
from time import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scannetv2 import ScanNetV2, scan_collate_fn
from utils.timm.scheduler.cosine_lr import CosineLRScheduler
from utils.timm.optim import create_optimizer_v2
import utils.util as util

# ------------------------
# 参数解析
# ------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="glipe")
parser.add_argument("--config", type=str, default="config")
parser.add_argument("--cur_id", type=str, default="0.2")
args = parser.parse_args()
MODEL = args.model
CONFIG = args.config
CUR_ID = args.cur_id

# 动态导入模块
MODEL_module = importlib.import_module(MODEL)
CONFIG_module = importlib.import_module(CONFIG)
print(args.model)
print(args.config)

# 路径设置
sys.path.append(str(Path(__file__).absolute().parent.parent))
print(os.path.basename(__file__))

# ------------------------
# 配置读取
# ------------------------
GLiPE = getattr(MODEL_module, "GLiPE")
scan_args = getattr(CONFIG_module, "scan_args")
scan_warmup_args = getattr(CONFIG_module, "scan_warmup_args")
glipe_args = getattr(CONFIG_module, "glipe_args")
batch_size = getattr(CONFIG_module, "batch_size")
lr = getattr(CONFIG_module, "learning_rate")
ls = getattr(CONFIG_module, "label_smoothing")
epoch = getattr(CONFIG_module, "epoch")
warmup = getattr(CONFIG_module, "warmup")

cur_id = CUR_ID
torch.set_float32_matmul_precision("high")

# ------------------------
# warmup 函数
# ------------------------
def warmup_fn(model, dataset, optimizer):
    model.train()
    traindlr = DataLoader(dataset, batch_size=len(dataset), collate_fn=scan_collate_fn, pin_memory=True, num_workers=6)
    for xyz, feature, indices, pts, y in traindlr:
        xyz = xyz.cuda(non_blocking=True)
        feature = feature.cuda(non_blocking=True)
        indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
        pts = pts.tolist()[::-1]
        y = y.cuda(non_blocking=True)

        # 前向 + 反向
        p, closs = model(xyz, feature, indices, pts)
        loss = F.cross_entropy(p, y, label_smoothing=ls, ignore_index=20) + closs
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

# ------------------------
# 日志和目录
# ------------------------
os.makedirs(f"output/log/{cur_id}", exist_ok=True)
os.makedirs(f"output/model/{cur_id}", exist_ok=True)
logfile = open(f"output/log/{cur_id}/out.log", "a", 1)
errfile = open(f"output/log/{cur_id}/err.log", "a", 1)
sys.stdout = logfile
sys.stderr = errfile

# ------------------------
# 数据加载
# ------------------------
traindlr = DataLoader(
    ScanNetV2(scan_args, partition="train", loop=6),
    batch_size=batch_size,
    collate_fn=scan_collate_fn,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
    num_workers=16
)
testdlr = DataLoader(
    ScanNetV2(scan_args, partition="val", loop=1, train=False),
    batch_size=1,
    collate_fn=scan_collate_fn,
    pin_memory=True,
    persistent_workers=True,
    num_workers=16
)

step_per_epoch = len(traindlr)

# ------------------------
# 模型与优化器
# ------------------------
model = GLiPE(glipe_args).cuda()
optimizer = create_optimizer_v2(model, lr=lr, weight_decay=5e-2)
scheduler = CosineLRScheduler(
    optimizer,
    t_initial=epoch * step_per_epoch,
    lr_min=lr / 10000,
    warmup_t=warmup * step_per_epoch,
    warmup_lr_init=lr / 20
)

# ------------------------
# 继续训练配置
# ------------------------
resume = False
if resume:
    start_epoch = util.load_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer)["start_epoch"]
else:
    start_epoch = 0

scheduler_step = start_epoch * step_per_epoch

# ------------------------
# 训练指标
# ------------------------
metric = util.Metric(20)
ttls = util.AverageMeter()
corls = util.AverageMeter()
best = 0

# ------------------------
# 预热
# ------------------------
warmup_fn(model, ScanNetV2(scan_warmup_args, partition="train", loop=batch_size, warmup=True), optimizer)

# ------------------------
# 训练循环
# ------------------------
for i in range(start_epoch, epoch):
    model.train()
    ttls.reset()
    metric.reset()
    corls.reset()
    now = time()

    for xyz, feature, indices, pts, y in traindlr:
        lam = scheduler_step / (epoch * step_per_epoch)
        lam = 3e-3 ** lam * 0.2
        scheduler.step(scheduler_step)
        scheduler_step += 1

        xyz = xyz.cuda(non_blocking=True)
        feature = feature.cuda(non_blocking=True)
        indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
        pts = pts.tolist()[::-1]
        y = y.cuda(non_blocking=True)
        mask = y != 20

        # 前向 + 反向
        p, closs = model(xyz, feature, indices, pts)
        loss = F.cross_entropy(p, y, label_smoothing=ls, ignore_index=20) + lam * closs

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        metric.update(p.detach()[mask], y[mask])
        ttls.update(loss.item())
        corls.update(closs.item())

    # 输出训练信息
    print(f"epoch {i}:")
    print(f"loss: {round(ttls.avg, 4)} || cls: {round(corls.avg, 4)}")
    metric.print("train:")

    # 验证
    model.eval()
    metric.reset()
    with torch.no_grad():
        for xyz, feature, indices, pts, y in testdlr:
            xyz = xyz.cuda(non_blocking=True)
            feature = feature.cuda(non_blocking=True)
            indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
            y = y.cuda(non_blocking=True)
            mask = y != 20

            p = model(xyz, feature, indices)
            metric.update(p[mask], y[mask])

    metric.print("val:  ")
    print(f"duration: {time() - now}")
    cur = metric.miou
    if best < cur:
        best = cur
        print("new best!")
        util.save_state(f"output/model/{cur_id}/best.pt", model=model)

    # 保存最后状态
    util.save_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer,
                    start_epoch=i + 1)
