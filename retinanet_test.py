# test_retinanet.py — final Test 评估（不改训练文件）
import argparse, time, numpy as np, torch, torch.utils.data as data, torchvision as tv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from torchvision.ops import box_iou

from retinanet_train import (
    build_retinanet, YoloTxtWrapper, ResizeShortSide, collate_fn, set_seed, compute_map50
)

def load_state_dict_safely(path):
    """兼容两种保存：纯 state_dict 或 {'model': state_dict, ...}"""
    try:
        sd = torch.load(path, map_location='cpu', weights_only=True)  # torch>=2.4
        if isinstance(sd, dict) and any(k.startswith('head') or 'backbone' in k for k in sd.keys()):
            return sd
    except TypeError:
        pass
    ckpt = torch.load(path, map_location='cpu')
    return ckpt.get('model', ckpt)

def match_tp_pairs(pred, gt, iou_thr=0.5):
    """
    贪心匹配 TP:按分数高到低遍历预测,和尚未匹配的 GT 做 IoU 最大匹配。
    返回：列表[(gt_label, pred_label, pred_score)]
    """
    pb, ps, pl = pred["boxes"], pred["scores"], pred["labels"]
    gb, gl = gt["boxes"], gt["labels"]
    if len(pb) == 0 or len(gb) == 0:
        return []

    ious = box_iou(pb, gb)  # [Np, Ng]
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
    ap.add_argument('--score-thresh', type=float, default=0.05)  # 可调 0.0~0.5
    ap.add_argument('--iou-thr', type=float, default=0.5)       # 统一 IoU 阈值
    args = ap.parse_args()

    set_seed(42)
    device = torch.device(args.device)

    # === dataset / loader ===
    resize = ResizeShortSide(args.short_side, args.max_size)
    ds = YoloTxtWrapper(
        args.test_images, args.test_labels,
        transforms=tv.transforms.Compose([resize]),
        num_classes=args.num_classes
    )
    loader = data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    # === model ===
    model = build_retinanet(num_classes=args.num_classes).to(device)
    state = load_state_dict_safely(args.ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    # 调分数阈值（RetinaNet 的后处理）
    # torchvision 0.20 的 RetinaNet 暴露 model.score_thresh
    if hasattr(model, "score_thresh"):
        model.score_thresh = float(args.score_thresh)
    elif hasattr(model, "head") and hasattr(model.head, "score_thresh"):
        model.head.score_thresh = float(args.score_thresh)

    # === forward (collect preds / gts) ===
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

    # === detection metric: mAP@0.5 ===
    det = compute_map50(all_preds, all_gts, iou_thr=args.iou_thr, num_classes=args.num_classes)
    mAP50 = det["mAP50"]

    # === classification metrics on matched TP pairs ===
    y_true, y_pred, score_rows = [], [], []
    for p, g in zip(all_preds, all_gts):
        pairs = match_tp_pairs(p, g, iou_thr=args.iou_thr)
        for gt_lab, pr_lab, pr_score in pairs:
            y_true.append(gt_lab)
            y_pred.append(pr_lab)
            # 仅给预测类打分，其它类置 0（简化版 OVR 分数）
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

    # === print ===
    print("\n=== TEST RESULTS ===")
    print(f"mAP@{args.iou_thr:.2f}:     {mAP50:.4f}")
    print(f"Precision (macro): {P:.4f}")
    print(f"Recall    (macro): {R:.4f}")
    print(f"F1        (macro): {F1:.4f}")
    print(f"Accuracy:          {ACC:.4f}")
    print(f"ROC-AUC (ovr):     {AUC:.4f}")
    print(f"Eval time:         {test_time:.2f}s\n")

if __name__ == "__main__":
    main()
