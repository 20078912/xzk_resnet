# visualize_retinanet.py — 预测可视化 + 成功/失败样本展示 + per-class 统计

import os
import shutil
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import nms
from retinanet_train import build_retinanet, YoloTxtWrapper, ResizeShortSide, set_seed

# ====================== 配置 ======================
CKPT = "outputs_retinanet/best.pth"
TEST_IMG = "dataset/test/images"
TEST_LBL = "dataset/test/labels"
NUM_CLASSES = 12
OUT_DIR = "vis_results"
CONF_THRESH = 0.2
IOU_THR = 0.5
PER_CLASS = 3           # 每类展示多少成功/失败样本
MAX_BOX_PER_IMG = 5     # 每张图最多画几个预测框

# ====================== 类别自动检测 ======================
def infer_class_ids(labels_dir):
    ids = set()
    for txt in Path(labels_dir).glob("*.txt"):
        with open(txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        ids.add(int(float(parts[0])))
                    except ValueError:
                        pass
    return sorted(list(ids))

REAL_CLASS_IDS = infer_class_ids(TEST_LBL)
print("✅ Detected label IDs from test labels:", REAL_CLASS_IDS)

# 手动定义类别名（与你训练时一致）
CLASS_NAMES = [
    "ant", "bee", "beetle", "butterfly", "caterpillar", "dragonfly",
    "fly", "grasshopper", "mosquito", "moth", "spider", "wasp"
]
ID2NAME = {i: CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"cls{i}" for i in REAL_CLASS_IDS}

# ====================== 初始化目录 ======================
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(f"{OUT_DIR}/success", exist_ok=True)
os.makedirs(f"{OUT_DIR}/failure", exist_ok=True)

# ====================== IoU计算 ======================
def box_iou(box1, box2):
    if len(box1) == 0 or len(box2) == 0:
        return torch.zeros((len(box1), len(box2)))
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (area1[:, None] + area2 - inter + 1e-6)

# ====================== 主逻辑 ======================
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🔍 Running RetinaNet inference on ALL test samples...")

model = build_retinanet(num_classes=NUM_CLASSES).to(device)
ckpt = torch.load(CKPT, map_location=device)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

resize = ResizeShortSide(short_side=800, max_size=1333)
ds = YoloTxtWrapper(TEST_IMG, TEST_LBL, transforms=resize, num_classes=NUM_CLASSES)

success_per_class = {i: [] for i in REAL_CLASS_IDS}
failure_per_class = {i: [] for i in REAL_CLASS_IDS}
success_count = np.zeros(NUM_CLASSES)
total_count = np.zeros(NUM_CLASSES)

for i in range(len(ds)):  # 遍历所有 test 样本
    img, target = ds[i]
    img_vis = (img * 255).byte()
    img_t = img.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_t)[0]

    boxes, scores, labels = preds["boxes"].cpu(), preds["scores"].cpu(), preds["labels"].cpu()
    keep = scores >= CONF_THRESH
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if len(boxes) > 0:
        nms_keep = nms(boxes, scores, iou_threshold=0.5)
        boxes, scores, labels = boxes[nms_keep][:MAX_BOX_PER_IMG], scores[nms_keep][:MAX_BOX_PER_IMG], labels[nms_keep][:MAX_BOX_PER_IMG]

    gtb, gtl = target["boxes"], target["labels"]
    max_iou = 0.0
    if len(boxes) > 0 and len(gtb) > 0:
        ious = box_iou(boxes, gtb)
        max_iou = ious.max().item()
    is_success = max_iou >= IOU_THR

    for gt in gtl.tolist():
        total_count[gt] += 1
        if is_success:
            success_count[gt] += 1

    label_texts = [f"{ID2NAME.get(int(l), str(l))} {s:.2f}" for l, s in zip(labels, scores)]
    if len(boxes) > 0:
        drawn = draw_bounding_boxes(img_vis, boxes, labels=label_texts, colors="lime", width=2)
    else:
        drawn = img_vis
    img_pil = F.to_pil_image(drawn)

    fname = Path(ds.img_paths[i]).stem
    tag = "success" if is_success else "failure"
    img_pil.save(f"{OUT_DIR}/{tag}/{fname}.jpg")

    for gt in gtl.tolist():
        if is_success and len(success_per_class[gt]) < PER_CLASS:
            success_per_class[gt].append(img_pil)
        elif not is_success and len(failure_per_class[gt]) < PER_CLASS:
            failure_per_class[gt].append(img_pil)

# ====================== 拼接整合图 ======================
def make_class_grid(class_dict, title, outfile):
    fig, axes = plt.subplots(len(class_dict), PER_CLASS, figsize=(PER_CLASS*3, len(class_dict)*3))
    fig.suptitle(title, fontsize=16)
    if len(class_dict) == 1:
        axes = np.expand_dims(axes, 0)
    for row_idx, cls_id in enumerate(class_dict.keys()):
        imgs = class_dict[cls_id]
        for j in range(PER_CLASS):
            ax = axes[row_idx, j] if len(class_dict) > 1 else axes[j]
            if j < len(imgs):
                ax.imshow(imgs[j])
            ax.axis("off")
        axes[row_idx, 0].set_ylabel(ID2NAME.get(cls_id, f"cls{cls_id}"),
                                    rotation=0, labelpad=40, fontsize=12, color="blue")
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"✅ Saved: {outfile}")

print("\n✅ Inference complete. Now generating summary grids...\n")
make_class_grid(success_per_class, "Success Cases", f"{OUT_DIR}/summary_success.jpg")
make_class_grid(failure_per_class, "Failure Cases", f"{OUT_DIR}/summary_failure.jpg")

# ====================== 输出统计 ======================
csv_path = f"{OUT_DIR}/results_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "Total", "Success", "Accuracy (%)"])
    for i in REAL_CLASS_IDS:
        total = int(total_count[i])
        succ = int(success_count[i])
        rate = succ / total * 100 if total > 0 else 0
        writer.writerow([ID2NAME[i], total, succ, f"{rate:.1f}"])

print(f"\n📊 Detection Summary (also saved to {csv_path}):")
for i in REAL_CLASS_IDS:
    name = ID2NAME[i]
    total = int(total_count[i])
    succ = int(success_count[i])
    rate = succ / total * 100 if total > 0 else 0
    print(f"{name:12s}: {succ}/{total} ({rate:.1f}%)")

print(f"\n🎯 Visualization done!\nCheck folders: {OUT_DIR}/success/ , {OUT_DIR}/failure/\nSummary grids saved in {OUT_DIR}/summary_*.jpg\n")
