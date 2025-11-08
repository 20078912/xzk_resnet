# test_retinanet.py — final Test 评估 + 可视化增强
import argparse, time, os, numpy as np, torch, torch.utils.data as data, torchvision as tv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from torchvision.ops import box_iou

from retinanet_train import (
    build_retinanet, YoloTxtWrapper, ResizeShortSide, collate_fn, set_seed, compute_map50
)


def load_state_dict_safely(path):
    """兼容纯 state_dict / checkpoint 两种格式"""
    try:
        sd = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(sd, dict) and any(k.startswith('head') or 'backbone' in k for k in sd.keys()):
            return sd
    except TypeError:
        pass
    ckpt = torch.load(path, map_location='cpu')
    return ckpt.get('model', ckpt)


def match_tp_pairs(pred, gt, iou_thr=0.5):
    """匹配预测框和真实框（贪心方式）"""
    pb, ps, pl = pred["boxes"], pred["scores"], pred["labels"]
    gb, gl = gt["boxes"], gt["labels"]
    if len(pb) == 0 or len(gb) == 0:
        return []

    ious = box_iou(pb, gb)
    order = torch.argsort(ps, descending=True)
    used_g = set()
    pairs = []
    for i in order.tolist():
        if gb.numel() == 0:
            break
        j = int(torch.argmax(ious[i]))
        if float(ious[i, j]) >= iou_thr and j not in used_g:
            used_g.add(j)
            pairs.append((int(gl[j]), int(pl[i]), float(ps[i])))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--test-images', required=True)
    ap.add_argument('--test-labels', required=True)
    ap.add_argument('--num-classes', type=int, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--short-side', type=int, default=800)
    ap.add_argument('--max-size', type=int, default=1333)
    ap.add_argument('--score-thresh', type=float, default=0.05)
    ap.add_argument('--iou-thr', type=float, default=0.5)
    ap.add_argument('--out-dir', type=str, default='outputs_retinanet_test')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(42)
    device = torch.device(args.device)

    # === Dataset ===
    resize = ResizeShortSide(args.short_side, args.max_size)
    ds = YoloTxtWrapper(
        args.test_images, args.test_labels,
        transforms=tv.transforms.Compose([resize]),
        num_classes=args.num_classes
    )
    loader = data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    # === Model ===
    model = build_retinanet(num_classes=args.num_classes).to(device)
    state = load_state_dict_safely(args.ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    if hasattr(model, "score_thresh"):
        model.score_thresh = float(args.score_thresh)

    # === Forward ===
    all_preds, all_gts = [], []
    t0 = time.time()
    with torch.no_grad():
        for images, targets in loader:
            images = [im.to(device) for im in images]
            outputs = model(images)
            for o, t in zip(outputs, targets):
                all_preds.append({
                    "boxes": o["boxes"].detach().cpu(),
                    "scores": o["scores"].detach().cpu(),
                    "labels": o["labels"].detach().cpu()
                })
                all_gts.append({
                    "boxes": t["boxes"],
                    "labels": t["labels"]
                })
    test_time = time.time() - t0

    # === Detection metrics ===
    det = compute_map50(all_preds, all_gts, iou_thr=args.iou_thr, num_classes=args.num_classes)
    mAP50 = det["mAP50"]
    ap_per_class = det.get("AP50_per_class", {})

    # === Classification metrics ===
    y_true, y_pred, score_rows = [], [], []
    for p, g in zip(all_preds, all_gts):
        pairs = match_tp_pairs(p, g, iou_thr=args.iou_thr)
        for gt_lab, pr_lab, pr_score in pairs:
            y_true.append(gt_lab)
            y_pred.append(pr_lab)
            row = np.zeros(args.num_classes, dtype=float)
            row[pr_lab] = pr_score
            score_rows.append(row)

    if len(y_true) == 0:
        P = R = F1 = ACC = 0.0
        AUC = float('nan')
    else:
        P, R, F1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        ACC = accuracy_score(y_true, y_pred)
        try:
            y_true_oh = np.eye(args.num_classes, dtype=float)[y_true]
            scores = np.vstack(score_rows)
            AUC = float(roc_auc_score(y_true_oh, scores, multi_class="ovr"))
        except Exception:
            AUC = float('nan')

    # === Print summary ===
    print("\n=== TEST RESULTS ===")
    print(f"mAP@{args.iou_thr:.2f}:     {mAP50:.4f}")
    print(f"Precision (macro): {P:.4f}")
    print(f"Recall    (macro): {R:.4f}")
    print(f"F1        (macro): {F1:.4f}")
    print(f"Accuracy:          {ACC:.4f}")
    print(f"ROC-AUC (ovr):     {AUC:.4f}")
    print(f"Eval time:         {test_time:.2f}s\n")

    # === 每类结果导出 ===
    class_ids = list(range(args.num_classes))
    ap_list = [ap_per_class.get(c, 0.0) for c in class_ids]

    df = pd.DataFrame({
        "class_id": class_ids,
        "AP50": ap_list
    })

    # 如果有匹配样本，统计 per-class PRF
    if len(y_true) > 0:
        pcls, rcls, fcls, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=class_ids, zero_division=0
        )
        df["Precision"] = pcls
        df["Recall"] = rcls
        df["F1"] = fcls
        df["Support"] = sup
    else:
        df["Precision"] = df["Recall"] = df["F1"] = 0.0
        df["Support"] = 0

    csv_path = os.path.join(args.out_dir, "class_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Per-class results saved to {csv_path}")

    # === 绘制 per-class AP 柱状图 ===
    plt.figure(figsize=(10, 5))
    plt.bar(df["class_id"], df["AP50"], color="skyblue")
    plt.xlabel("Class ID"); plt.ylabel("AP@0.5")
    plt.title("Per-Class Average Precision (AP50)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(df["class_id"])
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "per_class_map.png")
    plt.savefig(fig_path, dpi=300)
    print(f"✅ Saved per-class AP plot: {fig_path}")


if __name__ == "__main__":
    main()
