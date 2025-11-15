# test_retinanet_final.py
import argparse, os, time
import numpy as np
import torch
import torch.utils.data as data
import torchvision as tv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score
)
from torchvision.ops import box_iou

# 从训练文件导入所有稳定模块
from retinanet_train_final import (
    build_retinanet,
    YoloTxtWrapper,
    ResizeShortSide,
    collate_fn,
    evaluate,            # ⭐ 使用训练版本 evaluate() 计算 detection 指标
    set_seed
)

# -----------------------
# 匹配 TP 用于分类指标
# -----------------------
def match_tp_pairs(pred, gt, iou_thr=0.5):
    pb, ps, pl = pred["boxes"], pred["scores"], pred["labels"]
    gb, gl = gt["boxes"], gt["labels"]

    if len(pb) == 0 or len(gb) == 0:
        return []

    ious = box_iou(pb, gb)
    order = torch.argsort(ps, descending=True)

    used_g = set()
    pairs = []

    for i in order.tolist():
        j = int(torch.argmax(ious[i]))
        if float(ious[i, j]) >= iou_thr and j not in used_g:
            used_g.add(j)
            pairs.append((int(gl[j]), int(pl[i]), float(ps[i])))

    return pairs


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test-images", required=True)
    ap.add_argument("--test-labels", required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--out-dir", default="outputs_retinanet_test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(42)
    device = torch.device(args.device)

    # -----------------------
    # Dataset (same as train)
    # -----------------------
    resize = ResizeShortSide(800, 1333)

    ds = YoloTxtWrapper(
        args.test_images,
        args.test_labels,
        transforms=tv.transforms.Compose([resize]),
        num_classes=args.num_classes
    )

    loader = data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # -----------------------
    # Load model
    # -----------------------
    model = build_retinanet(args.num_classes).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # -----------------------
    # Detection metrics
    # -----------------------
    metrics, det_time = evaluate(model, loader, device, args.num_classes)

    # -----------------------
    # For Classification metrics
    # -----------------------
    preds, gts = [], []
    with torch.no_grad():
        for imgs, tgts in loader:
            imgs = [i.to(device) for i in imgs]
            outs = model(imgs)
            for o, t in zip(outs, tgts):
                preds.append({
                    "boxes": o["boxes"].cpu(),
                    "scores": o["scores"].cpu(),
                    "labels": o["labels"].cpu()
                })
                gts.append({
                    "boxes": t["boxes"],
                    "labels": t["labels"]
                })

    # -----------------------
    # Classification metrics
    # -----------------------
    y_true, y_pred, score_rows = [], [], []

    for p, g in zip(preds, gts):
        pairs = match_tp_pairs(p, g, iou_thr=args.iou_thr)
        for gt_c, pr_c, sc in pairs:
            y_true.append(gt_c)
            y_pred.append(pr_c)

            row = np.zeros(args.num_classes)
            row[pr_c] = sc
            score_rows.append(row)

    if len(y_true) == 0:
        cls_P = cls_R = cls_F1 = cls_ACC = 0.0
        cls_AUC = float("nan")
    else:
        cls_P, cls_R, cls_F1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        cls_ACC = accuracy_score(y_true, y_pred)
        try:
            y_true_oh = np.eye(args.num_classes)[y_true]
            scores = np.vstack(score_rows)
            cls_AUC = float(roc_auc_score(y_true_oh, scores, multi_class="ovr"))
        except:
            cls_AUC = float("nan")

    # -----------------------
    # Output summary
    # -----------------------
    print("\n===== TEST RESULTS =====")
    print(f"mAP@0.50:        {metrics['mAP50']:.4f}")
    print(f"mAP@0.50:0.95:   {metrics['mAP5095']:.4f}")
    print(f"Detection P:     {metrics['precision']:.4f}")
    print(f"Detection R:     {metrics['recall']:.4f}")
    print(f"[Cls] Precision: {cls_P:.4f}")
    print(f"[Cls] Recall:    {cls_R:.4f}")
    print(f"[Cls] F1:        {cls_F1:.4f}")
    print(f"[Cls] Accuracy:  {cls_ACC:.4f}")
    print(f"[Cls] ROC-AUC:   {cls_AUC:.4f}")
    print(f"Eval time:       {det_time:.2f}s")

    # -----------------------
    # Per-class AP saving
    # -----------------------
    class_ids = list(range(args.num_classes))

    CLASS_NAMES = [
        "ant", "bee", "beetle", "butterfly", "caterpillar", "dragonfly",
        "fly", "grasshopper", "mosquito", "moth", "spider", "wasp"
    ][:args.num_classes]

    df = pd.DataFrame({
        "class_id": class_ids,
        "class_name": CLASS_NAMES,
        "AP50": [metrics["AP50_per_class"][c] for c in class_ids]
    })


    df.to_csv(os.path.join(args.out_dir, "class_results.csv"), index=False)
    print(f"\nSaved per-class AP results → {args.out_dir}/class_results.csv")

    plt.figure(figsize=(14, 7))
    plt.bar(df["class_name"], df["AP50"])
    plt.xticks(df["class_name"], rotation=45, ha="right")  # 强制使用类别名
    plt.xlabel("Class Name")  # 可选
    plt.ylabel("AP@50")        # 可选
    plt.title("Per-class AP@0.50")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "per_class_map.png"), dpi=300)

    print(f"Saved plot → {args.out_dir}/per_class_map.png")


if __name__ == "__main__":
    main()
