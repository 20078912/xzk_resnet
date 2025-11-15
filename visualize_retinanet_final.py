# visualize_retinanet_final.py
# 完整可视化：成功/失败样本、框、类别名、per-class grid、CSV统计

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

from retinanet_train_final import (
    build_retinanet,
    YoloTxtWrapper,
    ResizeShortSide,
    set_seed
)

# ----------------------------
# 用户可调参数
# ----------------------------
CKPT = "outputs_retinanet_final/best.pth"
TEST_IMG = "dataset/test/images"
TEST_LBL = "dataset/test/labels"
NUM_CLASSES = 12

OUT_DIR = "vis_results_final"

CONF_THRESH = 0.25
IOU_THR = 0.5
PER_CLASS = 3
MAX_BOX_PER_IMG = 5

# 类别名（和训练一致）
CLASS_NAMES = [
    "ant","bee","beetle","butterfly","caterpillar","dragonfly",
    "fly","grasshopper","mosquito","moth","spider","wasp"
]


# ----------------------------
# 自定义 IoU（与旧版本对齐）
# ----------------------------
def box_iou_local(box1, box2):
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


# ----------------------------
# 自动检测 test labels 里的类别
# ----------------------------
def infer_class_ids(labels_dir):
    ids = set()
    for txt in Path(labels_dir).glob("*.txt"):
        with open(txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    ids.add(int(float(parts[0])))
    return sorted(list(ids))


# ----------------------------
# 主流程
# ----------------------------
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 类别 ID
    REAL_CLASS_IDS = infer_class_ids(TEST_LBL)
    ID2NAME = {i: CLASS_NAMES[i] for i in REAL_CLASS_IDS}

    # 创建输出目录
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(f"{OUT_DIR}/success", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/failure", exist_ok=True)

    # 载入模型
    print("⚡ Loading model...")
    model = build_retinanet(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # dataset
    resize = ResizeShortSide(800, 1333)
    ds = YoloTxtWrapper(TEST_IMG, TEST_LBL, transforms=resize, num_classes=NUM_CLASSES)

    success_per_class = {i: [] for i in REAL_CLASS_IDS}
    failure_per_class = {i: [] for i in REAL_CLASS_IDS}
    success_count = np.zeros(NUM_CLASSES)
    total_count = np.zeros(NUM_CLASSES)

    print(f"🔍 Processing {len(ds)} images...")

    for idx in range(len(ds)):
        if idx % 50 == 0:
            print(f"  {idx}/{len(ds)}")

        img, gt = ds[idx]
        img_vis = (img * 255).byte()
        img_t = img.unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            pred = model(img_t)[0]

        boxes = pred["boxes"].cpu()
        scores = pred["scores"].cpu()
        labels = pred["labels"].cpu()

        # 过滤低分
        keep = scores >= CONF_THRESH
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # NMS
        if len(boxes) > 0:
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            scores = scores[valid]
            labels = labels[valid]

            if len(boxes) > 0:
                keep_nms = nms(boxes, scores, 0.5)
                boxes = boxes[keep_nms][:MAX_BOX_PER_IMG]
                scores = scores[keep_nms][:MAX_BOX_PER_IMG]
                labels = labels[keep_nms][:MAX_BOX_PER_IMG]

        # GT
        gtb = gt["boxes"]
        gtl = gt["labels"]

        # IoU 计算
        max_iou = 0
        if len(boxes) > 0 and len(gtb) > 0:
            max_iou = float(box_iou_local(boxes, gtb).max())

        is_success = max_iou >= IOU_THR

        # 统计 per-class
        for cls in gtl.tolist():
            total_count[cls] += 1
            if is_success:
                success_count[cls] += 1

        # 可视化
        label_text = [f"{ID2NAME[int(l)]} {float(s):.2f}" for l, s in zip(labels, scores)]

        if len(boxes) > 0:
            drawn = draw_bounding_boxes(img_vis, boxes, labels=label_text, colors="lime", width=2)
        else:
            drawn = img_vis

        img_pil = F.to_pil_image(drawn)
        fname = Path(ds.img_paths[idx]).stem

        tag = "success" if is_success else "failure"
        img_pil.save(f"{OUT_DIR}/{tag}/{fname}.jpg")

        # 收集 per-class 示例图
        for cls in gtl.tolist():
            if is_success and len(success_per_class[cls]) < PER_CLASS:
                success_per_class[cls].append(img_pil.copy())
            elif not is_success and len(failure_per_class[cls]) < PER_CLASS:
                failure_per_class[cls].append(img_pil.copy())

    print("🎉 Inference finished!\nGenerating grids...")

    # ----------------------------
    # 拼接 per-class 图
    # ----------------------------
    def make_grid(class_dict, title, outfile):
        valid = {k: v for k, v in class_dict.items() if len(v) > 0}
        if not valid:
            print(f"⚠ No samples for {title}")
            return

        rows = len(valid)
        fig, axes = plt.subplots(rows, PER_CLASS, figsize=(PER_CLASS * 3, rows * 3))
        fig.suptitle(title, fontsize=16)

        if rows == 1:
            axes = np.expand_dims(axes, 0)

        for r, (cls, imgs) in enumerate(valid.items()):
            for c in range(PER_CLASS):
                ax = axes[r, c]
                if c < len(imgs):
                    ax.imshow(imgs[c])
                ax.axis("off")
            axes[r, 0].set_ylabel(CLASS_NAMES[cls], rotation=0, labelpad=40, fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(outfile, dpi=200)
        plt.close()

        print(f"✔ Saved {outfile}")

    make_grid(success_per_class, "Success Samples", f"{OUT_DIR}/summary_success.jpg")
    make_grid(failure_per_class, "Failure Samples", f"{OUT_DIR}/summary_failure.jpg")

    # ----------------------------
    # 保存 per-class 成功率 CSV
    # ----------------------------
    csv_path = f"{OUT_DIR}/results_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Class", "Total", "Success", "Accuracy %"])
        for cls in REAL_CLASS_IDS:
            t = int(total_count[cls])
            s = int(success_count[cls])
            acc = s / t * 100 if t > 0 else 0
            w.writerow([CLASS_NAMES[cls], t, s, f"{acc:.1f}"])

    print(f"\n📊 Summary saved to {csv_path}")
    print(f"\nImages saved under: {OUT_DIR}/success/ and {OUT_DIR}/failure/")
    print("🎯 DONE!")


if __name__ == "__main__":
    main()
