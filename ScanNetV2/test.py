import torch
from torch import nn
import torch.nn.functional as F
from scannetv2 import ScanNetV2, scan_test_collate_fn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import utils.util as util
from glipe_sim import GLiPE
from config import scan_args, glipe_args
from torch.amp import autocast

torch.set_float32_matmul_precision("high")

# loop x rotation x scaling
loop = 4 * 4 * 3

testdlr = DataLoader(ScanNetV2(scan_args, partition="val", loop=loop, train=False, test=True), batch_size=1,
                      collate_fn=scan_test_collate_fn, pin_memory=True, num_workers=16)

model = GLiPE(glipe_args).cuda()

util.load_state("/home/yy/01PointGLiPE/ScanNetV2/output/model/00.1/best.pt", model=model)

model.eval()

metric = util.Metric(20)
cum = 0
cnt = 0

with torch.no_grad():
    for xyz, feature, indices, nn, y in testdlr:
            xyz = xyz.cuda(non_blocking=True)
            feature = feature.cuda(non_blocking=True)
            indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
            nn = nn.cuda(non_blocking=True).long()
            with autocast('cuda'):
                p = model(xyz, feature, indices)
            cum = cum + p[nn]
            cnt += 1
            if cnt % loop == 0:
                y = y.cuda(non_blocking=True)
                mask = y != 20
                metric.update(cum[mask], y[mask])
                cnt = cum = 0

metric.print("test: ")
