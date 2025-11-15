import torch
from torch.amp import autocast
from ModelNet40 import ModelNet40
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import utils.util as util
from glipe12 import GLiPE
from config import glipe_args, batch_size
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
torch.set_float32_matmul_precision("high")

testdlr = DataLoader(ModelNet40(partition="test"), batch_size=batch_size,
                      pin_memory=True, num_workers=6)

model = GLiPE(glipe_args).cuda()
util.load_state("/home/yy/01PointGLiPE/ModelNet40/output/model/12.2/best.pt", model=model)

metric = util.Metric(glipe_args.num_classes)


classnames = [
    "airplane","bathtub","bed","bench","bookshelf","bottle","bowl","car","chair",
    "cone","cup","curtain","desk","door","dresser","flower_pot","glass_box",
    "guitar","keyboard","lamp","laptop","mantel","monitor","night_stand","person",
    "piano","plant","radio","range_hood","sink","sofa","stairs","stool","table",
    "tent","toilet","tv_stand","vase","wardrobe","xbox"
]


# 2. 收集所有预测结果和真实标签
# ------------------------------
all_preds = []
all_labels = []



model.eval()
metric.reset()
with torch.no_grad():
    for xyz, y in testdlr:
        xyz = xyz.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        with autocast('cuda'):
            p = model(xyz)
        metric.update(p, y)

        pred = torch.argmax(p, dim=1)  # (B,)
        all_preds.append(pred.cpu())
        all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # ------------------------------
    # 3. 计算混淆矩阵
    # ------------------------------
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(len(classnames)))

    # 每行（真实类）归一化，使得每行和为 1
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # ------------------------------
    # 4. 绘制热力图（改进版）
    # ------------------------------
    # 自定义从白到深蓝的 colormap
    cmap = LinearSegmentedColormap.from_list("white_to_blue", ["#ffffff", "#003366"])

    plt.figure(figsize=(14, 12))
    ax=sns.heatmap(
        cm_normalized.T[::-1, :],
        xticklabels=classnames,
        yticklabels=classnames[::-1],
        cmap=cmap,
        annot=False,
        fmt=".2f",
        cbar=True,
        vmin=0.0,
        vmax=1.0,
        linecolor='gray',      # 网格线颜色
        linewidths=0.5         # 网格线宽度
    )

    # 字体大小调整 + 标签翻转（X=Pred, Y=True）
    plt.xlabel("True Class", fontsize=20, labelpad=10)
    plt.ylabel("Predicted Class", fontsize=20, labelpad=10)
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(rotation=0, fontsize=20)
    # plt.title("Confusion Matrix", fontsize=18, pad=15)
    plt.tight_layout()
    plt.rcParams["font.family"] = "DejaVu Serif"
    cbar = ax.collections[0].colorbar
    # cbar.set_label("Normalized Value", fontsize=16)  # colorbar 标签
    cbar.ax.tick_params(labelsize=20)  # colorbar 刻度字体大小

    num_pred, num_true = cm_normalized.T[::-1, :].shape
    for i in range(num_pred):  # 纵轴（预测）
        for j in range(num_true):  # 横轴（真实）
            if cm_normalized.T[::-1, :][i, j] > 0.9:
                ax.text(
                    j + 0.5, i + 0.75, "*",  # +0.5使文本居中
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=20,
                    fontweight="bold"
                )

    # ------------------------------
    # 5. 保存到指定路径
    # ------------------------------
    save_dir = "/home/yy/01PointGLiPE/ModelNet40/output"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "confusion_matrix_white_to_blue.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

metric.print("val:  ", iou=False)
