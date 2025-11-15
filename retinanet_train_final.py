# retinanet_train_full_final.py
# Full-feature RetinaNet Training Script
# Supports: YOLO txt dataset, synthetic dataset, mAP50, mAP50-95, precision/recall,
#           train_loss, val_loss, scheduler, resume, best.pth, last.pth, plots

import argparse, os, time, random, csv
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torchvision as tv
from torchvision.ops import box_iou
from torch.amp import autocast, GradScaler
from PIL import Image as PILImage
# ---------------------
# Random Seed
# ---------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------
# Collate function
# ---------------------
def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


# ---------------------
# Clamp bounding boxes
# ---------------------
def clamp_boxes(boxes, w, h):
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h - 1)
    return boxes
# ---------------------
# Synthetic Detection Dataset
# ---------------------
class SyntheticDetectionDataset(data.Dataset):
    def __init__(self, n=64, img_size=512, num_classes=4):
        self.n = n
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        C, H, W = 3, self.img_size, self.img_size
        img = torch.zeros(C, H, W)
        num_obj = random.randint(1, 5)

        boxes = []
        labels = []

        for _ in range(num_obj):
            w = random.randint(20, 80)
            h = random.randint(20, 80)
            x1 = random.randint(0, W - w - 1)
            y1 = random.randint(0, H - h - 1)
            x2 = x1 + w
            y2 = y1 + h
            c = random.randint(0, self.num_classes - 1)

            boxes.append([x1, y1, x2, y2])
            labels.append(c)

            img[:, y1:y2, x1:x2] = torch.rand(3, 1, 1)

        return img, {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
# ---------------------
# YOLO TXT wrapper
# ---------------------
class YoloTxtWrapper(data.Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None, num_classes: Optional[int] = None):
        self.img_root = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self.num_classes = num_classes

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.img_paths = sorted([p for p in self.img_root.iterdir() if p.suffix.lower() in exts])

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def _yolo_to_xyxy(w, h, cx, cy, bw, bh):
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        return [x1, y1, x2, y2]

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = PILImage.open(img_path).convert("RGB")
        w, h = img.size

        txt_path = self.labels_dir / f"{img_path.stem}.txt"

        boxes = []
        labels = []

        if txt_path.exists():
            with open(txt_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    c, cx, cy, bw, bh = map(float, parts)
                    c = int(c)

                    if (self.num_classes is not None) and not (0 <= c < self.num_classes):
                        continue

                    boxes.append(self._yolo_to_xyxy(w, h, cx, cy, bw, bh))
                    labels.append(c)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4))
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = clamp_boxes(torch.tensor(boxes, dtype=torch.float32), w, h)
            labels = torch.tensor(labels, dtype=torch.int64)

        img_t = tv.transforms.functional.to_tensor(img)
        if self.transforms:
            img_t = self.transforms(img_t)

        return img_t, {"boxes": boxes, "labels": labels}
class ResizeShortSide:
    def __init__(self, short_side=800, max_size=1333):
        self.short = short_side
        self.max = max_size

    def __call__(self, img):
        C, H, W = img.shape
        s = self.short / min(H, W)
        newH, newW = int(round(H * s)), int(round(W * s))

        if max(newH, newW) > self.max:
            s2 = self.max / max(newH, newW)
            newH, newW = int(round(newH * s2)), int(round(newW * s2))

        return tv.transforms.functional.resize(img, [newH, newW])


# ---------------------
# Data Augmentation (Fast & Safe)
# ---------------------
train_aug = tv.transforms.Compose([
    tv.transforms.RandomHorizontalFlip(0.5),

    tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

    tv.transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),

    tv.transforms.RandomErasing(p=0.1)
])

# ---------------------
# mAP@0.50
# ---------------------
def compute_map50(preds: List[Dict], gts: List[Dict], iou_thr=0.5, num_classes=1):
    ap_per_class = {}
    eps = 1e-9

    for cls in range(num_classes):
        cls_preds = []
        cls_gts = []

        for p, g in zip(preds, gts):
            pm = (p["labels"] == cls)
            gm = (g["labels"] == cls)

            if pm.sum() == 0 and gm.sum() == 0:
                continue

            cls_preds.append({"boxes": p["boxes"][pm], "scores": p["scores"][pm]})
            cls_gts.append({"boxes": g["boxes"][gm]})

        flat = []
        for i, cp in enumerate(cls_preds):
            for b, s in zip(cp["boxes"], cp["scores"]):
                flat.append((i, float(s), b))

        flat.sort(key=lambda x: -x[1])

        tp = []
        fp = []
        matched = [set() for _ in cls_gts]
        total = sum(len(x["boxes"]) for x in cls_gts)

        for i, score, box in flat:
            g = cls_gts[i]["boxes"] if i < len(cls_gts) else torch.empty((0, 4))

            if len(g) == 0:
                fp.append(1)
                tp.append(0)
                continue

            ious = box_iou(box.unsqueeze(0), g).squeeze(0)
            j = int(torch.argmax(ious))
            v = float(ious[j])

            if v >= iou_thr and j not in matched[i]:
                tp.append(1)
                fp.append(0)
                matched[i].add(j)
            else:
                fp.append(1)
                tp.append(0)

        if total == 0:
            ap_per_class[cls] = 0.0
            continue

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        rec = tp / (total + eps)
        prec = tp / (tp + fp + eps)

        ap = 0.0
        for r in np.linspace(0, 1, 11):
            ap += prec[rec >= r].max() if np.any(rec >= r) else 0.0

        ap_per_class[cls] = ap / 11.0

    mAP = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    return {"AP50_per_class": ap_per_class, "mAP50": mAP}
# ---------------------
# mAP 50:95
# ---------------------
def compute_map5095(preds, gts, num_classes=1):
    iou_thresholds = np.arange(0.50, 0.96, 0.05)
    ap_table = {thr: {} for thr in iou_thresholds}

    for thr in iou_thresholds:
        r = compute_map50(preds, gts, iou_thr=thr, num_classes=num_classes)
        for cls, ap in r["AP50_per_class"].items():
            ap_table[thr][cls] = ap

    ap_per_class = {}

    for cls in range(num_classes):
        cls_aps = [ap_table[thr].get(cls, 0.0) for thr in iou_thresholds]
        ap_per_class[cls] = float(np.mean(cls_aps))

    map_5095 = float(np.mean(list(ap_per_class.values())))
    return {"AP5095_per_class": ap_per_class, "mAP5095": map_5095}
# ---------------------
# Precision / Recall
# ---------------------
def compute_precision_recall(preds, gts, iou_thr=0.5, num_classes=1):
    TP = 0
    FP = 0
    FN = 0

    for cls in range(num_classes):
        for p, g in zip(preds, gts):
            pm = (p["labels"] == cls)
            gm = (g["labels"] == cls)

            pb = p["boxes"][pm]
            gb = g["boxes"][gm]

            if len(gb) == 0:
                FP += len(pb)
                continue

            matched = set()
            for b in pb:
                ious = box_iou(b.unsqueeze(0), gb).squeeze(0)
                j = int(torch.argmax(ious))
                v = float(ious[j])

                if v >= iou_thr and j not in matched:
                    TP += 1
                    matched.add(j)
                else:
                    FP += 1

            FN += (len(gb) - len(matched))

    prec = TP / max(1, TP + FP)
    rec = TP / max(1, TP + FN)
    return prec, rec
# ---------------------
# Build RetinaNet
# ---------------------
def build_retinanet(num_classes=12, min_anchor=16):
    from torchvision.models import ResNet50_Weights

    model = tv.models.detection.retinanet_resnet50_fpn_v2(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        num_classes=num_classes,
    )

    sizes = []
    s = min_anchor
    for _ in range(5):
        sizes.append((s,))
        s *= 2

    model.anchor_generator.sizes = tuple(sizes)
    return model
# ---------------------
# Train one epoch
# ---------------------
def train_one_epoch(model, loader, optimizer, device, epoch, grad_accum, lr_scheduler, amp, scaler):
    model.train()
    loss_meter = 0.0
    t0 = time.time()

    for it, (images, targets) in enumerate(loader, 1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if amp:
            with autocast("cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values()) / grad_accum

            scaler.scale(loss).backward()

            if it % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values()) / grad_accum

            loss.backward()
            if it % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        loss_meter += float(loss.item()) * grad_accum
        # -------------------------
        # Progress Print  (every 20 iters)
        # -------------------------
        if it % 20 == 0 or it == 1:
            print(f"  Epoch {epoch} | Iter {it}/{len(loader)} | Loss={float(loss.item()):.4f}")


    if lr_scheduler:
        lr_scheduler.step()

    return loss_meter / max(1, len(loader)), time.time() - t0
# ---------------------
# Eval loss (FIXED)
# ---------------------
@torch.no_grad()
def eval_loss(model, loader, device):
    model.train()    
    total = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)  
        loss = sum(loss_dict.values()).item()
        total += loss

    model.eval()    
    return total / max(1, len(loader))


# ---------------------
# Evaluate
# ---------------------
@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    preds = []
    gts = []
    t0 = time.time()

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for o, t in zip(outputs, targets):
            preds.append({
                "boxes": o["boxes"].cpu(),
                "scores": o["scores"].cpu(),
                "labels": o["labels"].cpu(),
            })
            gts.append({
                "boxes": t["boxes"],
                "labels": t["labels"],
            })

    map50 = compute_map50(preds, gts, num_classes=num_classes)
    map5095 = compute_map5095(preds, gts, num_classes=num_classes)
    prec, rec = compute_precision_recall(preds, gts, num_classes=num_classes)

    return {
        "mAP50": map50["mAP50"],
        "mAP5095": map5095["mAP5095"],
        "precision": prec,
        "recall": rec,
        "AP50_per_class": map50["AP50_per_class"],
        "AP5095_per_class": map5095["AP5095_per_class"]
    }, time.time() - t0
# ---------------------
# Main
# ---------------------
def main():
    parser = argparse.ArgumentParser()

    # Dataset paths
    parser.add_argument("--train-images", type=str)
    parser.add_argument("--train-labels", type=str)
    parser.add_argument("--val-images", type=str)
    parser.add_argument("--val-labels", type=str)

    # Training config
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)

    # LR & resume
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["none", "step", "cosine"])

    # Output
    parser.add_argument("--out-dir", type=str, default="outputs_retinanet_final")

    args = parser.parse_args()
    set_seed(42)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    # ---------------------
    # Dataset
    # ---------------------
    resize = ResizeShortSide(800, 1333)

    train_ds = YoloTxtWrapper(
        args.train_images,
        args.train_labels,
        transforms=tv.transforms.Compose([resize, train_aug]),
        num_classes=args.num_classes
    )

    val_ds = YoloTxtWrapper(
        args.val_images,
        args.val_labels,
        transforms=tv.transforms.Compose([resize]),
        num_classes=args.num_classes
    )

    loader_settings = {
        "batch_size": args.batch_size,
        "collate_fn": collate_fn,
        "num_workers": args.workers,
        "pin_memory": (device.type == "cuda"),
    }

    if args.workers > 0:
        loader_settings["prefetch_factor"] = 2
        loader_settings["persistent_workers"] = True

    train_loader = data.DataLoader(
        train_ds,
        shuffle=True,
        **loader_settings
    )

    val_loader = data.DataLoader(
        val_ds,
        shuffle=False,
        **loader_settings
    )

    # ---------------------
    # Model / Optimizer
    # ---------------------
    model = build_retinanet(args.num_classes).to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, momentum=0.9, weight_decay=1e-4
    )

    scaler = GradScaler("cuda") if args.amp else None
    # ---------------------
    # Resume
    # ---------------------
    start_epoch = 0
    best_map50 = 0.0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and scaler:
            scaler.load_state_dict(ckpt["scaler"])

        start_epoch = ckpt.get("epoch", 0) + 1
        best_map50 = ckpt.get("best_map50", 0.0)

        print(f"🔄 Resumed from {args.resume} at epoch {start_epoch}, best mAP50={best_map50:.4f}")
    # ---------------------
    # Scheduler
    # ---------------------
    if args.scheduler == "none":
        lr_scheduler = None
    elif args.scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[args.epochs // 2, args.epochs * 3 // 4], gamma=0.1
        )
    elif args.scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    # ---------------------
    # Logging
    # ---------------------
    log_path = os.path.join(args.out_dir, "training_log.csv")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_loss",
            "mAP50", "mAP5095",
            "precision", "recall",
            "AP50_per_class", "AP5095_per_class",
            "train_time", "eval_time"
        ])
    # ---------------------
    # Training Loop
    # ---------------------
    for epoch in range(start_epoch, args.epochs):

        train_loss, train_time = train_one_epoch(
            model, train_loader, optimizer, device,
            epoch, args.grad_accum, lr_scheduler,
            args.amp, scaler
        )

        val_loss = eval_loss(model, val_loader, device)
        metrics, eval_time = evaluate(model, val_loader, device, args.num_classes)

        print(f"[Epoch {epoch}] "
              f"TL={train_loss:.4f}, VL={val_loss:.4f}, "
              f"mAP50={metrics['mAP50']:.4f}, "
              f"mAP95={metrics['mAP5095']:.4f}, "
              f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")

        # CSV Logging
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.4f}",
                f"{val_loss:.4f}",
                f"{metrics['mAP50']:.4f}",
                f"{metrics['mAP5095']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                str(metrics["AP50_per_class"]),
                str(metrics["AP5095_per_class"]),
                f"{train_time:.2f}",
                f"{eval_time:.2f}",
            ])
        # Save last
        last_ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "best_map50": best_map50,
        }
        torch.save(last_ckpt, os.path.join(args.out_dir, "last.pth"))

        # Save best
        if metrics["mAP50"] > best_map50:
            best_map50 = metrics["mAP50"]
            torch.save(last_ckpt, os.path.join(args.out_dir, "best.pth"))
            print(f"⭐ Saved BEST at epoch {epoch}: mAP50={best_map50:.4f}")
    # ---------------------
    # Plots
    # ---------------------
    df = pd.read_csv(log_path)

    # Loss curve
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.legend(); plt.grid(); plt.title("Loss Curve")
    plt.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=300)

    # mAP
    plt.figure()
    plt.plot(df["epoch"], df["mAP50"], label="mAP50")
    plt.plot(df["epoch"], df["mAP5095"], label="mAP50-95")
    plt.legend(); plt.grid(); plt.title("mAP Curve")
    plt.savefig(os.path.join(args.out_dir, "map_curve.png"), dpi=300)

    # Precision & Recall
    plt.figure()
    plt.plot(df["epoch"], df["precision"], label="Precision")
    plt.plot(df["epoch"], df["recall"], label="Recall")
    plt.legend(); plt.grid(); plt.title("Precision & Recall")
    plt.savefig(os.path.join(args.out_dir, "pr_curve.png"), dpi=300)
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()

