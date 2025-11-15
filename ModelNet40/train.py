import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="glipe30")
parser.add_argument("--config", type=str, default="config")
parser.add_argument("--cur_id", type=str, default="30.2")
args = parser.parse_args()
MODEL = args.model
CONFIG = args.config
CUR_ID = args.cur_id
print(args.model)
print(args.config)
import importlib

MODEL_module = importlib.import_module(MODEL)
CONFIG_module = importlib.import_module(CONFIG)

import torch
import torch.nn.functional as F
# removed mixed precision imports
# from torch.amp import autocast, GradScaler
from ModelNet40 import ModelNet40
from torch.utils.data import DataLoader
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.scheduler.cosine_lr import CosineLRScheduler
from utils.timm.optim import create_optimizer_v2
import utils.util as util
import os
from time import time
print(os.path.basename(__file__))
GLiPE = getattr(MODEL_module, "GLiPE")
glipe_args = getattr(CONFIG_module, "glipe_args")
batch_size = getattr(CONFIG_module, "batch_size")
lr = getattr(CONFIG_module, "learning_rate")
ls = getattr(CONFIG_module, "label_smoothing")
epoch = getattr(CONFIG_module, "epoch")
warmup = getattr(CONFIG_module, "warmup")

# keep as-is (does not enable amp by itself)
torch.set_float32_matmul_precision("high")

cur_id = CUR_ID
os.makedirs(f"output/log/{cur_id}", exist_ok=True)
os.makedirs(f"output/model/{cur_id}", exist_ok=True)
logfile = f"output/log/{cur_id}/out.log"
errfile = f"output/log/{cur_id}/err.log"
logfile = open(logfile, "a", 1)
errfile = open(errfile, "a", 1)
sys.stdout = logfile
sys.stderr = errfile

print(r"base")

traindlr = DataLoader(ModelNet40(), batch_size=batch_size,
                      shuffle=True, pin_memory=True,
                      persistent_workers=True, drop_last=True, num_workers=6)
testdlr = DataLoader(ModelNet40(partition="test"), batch_size=batch_size,
                     pin_memory=True,
                     persistent_workers=True, num_workers=6)

step_per_epoch = len(traindlr)

model = GLiPE(glipe_args).cuda()

optimizer = create_optimizer_v2(model, lr=lr, weight_decay=5e-2)
scheduler = CosineLRScheduler(optimizer, t_initial=epoch * step_per_epoch, lr_min=lr / 10000,
                              warmup_t=warmup * step_per_epoch, warmup_lr_init=lr / 20)

# removed scaler
# scalar = GradScaler(device='cuda')

# if wish to continue from a checkpoint
resume = False
if resume:
    # removed scalar from load_state call
    start_epoch = util.load_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer)["start_epoch"]
else:
    start_epoch = 0

scheduler_step = start_epoch * step_per_epoch

metric = util.Metric(glipe_args.num_classes)
ttls = util.AverageMeter()
best = 0
corls = util.AverageMeter()

for i in range(start_epoch, epoch):
    model.train()
    ttls.reset()
    corls.reset()
    metric.reset()
    now = time()
    for xyz, y in traindlr:
        scheduler.step(scheduler_step)
        scheduler_step += 1
        xyz = xyz.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        # removed autocast context; normal forward/backward
        p = model(xyz)
        loss = F.cross_entropy(p, y, label_smoothing=ls)

        metric.update(p.detach(), y)
        ttls.update(loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"epoch {i}:")
    print(f"loss: {round(ttls.avg, 4)} || cls: {round(corls.avg, 4)}")
    metric.print("train:", iou=False)

    model.eval()
    metric.reset()
    with torch.no_grad():
        for xyz, y in testdlr:
            xyz = xyz.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            # removed autocast here as well
            p = model(xyz)
            metric.update(p, y)

    metric.print("val:  ", iou=False)
    print(f"duration: {time() - now}")
    cur = metric.acc
    if best < cur:
        best = cur
        print("new best!")
        util.save_state(f"output/model/{cur_id}/best.pt", model=model)

    # removed scalar from save_state call
    util.save_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer, start_epoch=i + 1)
