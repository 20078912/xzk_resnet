# retinanet_train.py
import argparse, os, time, random
from pathlib import Path
from typing import List, Dict
import numpy as np

import torch
import torch.utils.data as data
import torchvision as tv
from torchvision.ops import box_iou
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights

# =========================
# Repro
# =========================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# =========================
# Dataloader utils
# =========================
def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)

def clamp_boxes(boxes, w, h):
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h - 1)
    return boxes

# =========================
# Wrap Transform (Windows-safe)
# =========================
class WrapTransformDataset(torch.utils.data.Dataset):
    def __init__(self, base, transform=None):
        self.base = base; self.transform = transform
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        img, tgt = self.base[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, tgt

# =========================
# Synthetic dataset (smoke test)
# =========================
class SyntheticDetectionDataset(data.Dataset):
    def __init__(self, n=64, img_size=512, num_classes=4):
        self.n=n; self.img_size=img_size; self.num_classes=num_classes
    def __len__(self): return self.n
    def __getitem__(self, idx):
        C,H,W = 3, self.img_size, self.img_size
        img = torch.zeros(C,H,W, dtype=torch.float32)
        num_obj = random.randint(1,5)
        boxes=[]; labels=[]
        for _ in range(num_obj):
            w = random.randint(20,80); h = random.randint(20,80)
            x1 = random.randint(0, W-w-1); y1 = random.randint(0, H-h-1)
            x2 = x1 + w; y2 = y1 + h
            c = random.randint(0, self.num_classes-1)
            boxes.append([x1,y1,x2,y2]); labels.append(c)
            img[:, y1:y2, x1:x2] = torch.rand(3,1,1)
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32),
                  "labels": torch.tensor(labels, dtype=torch.int64)}
        return img, target

# =========================
# COCO wrapper (optional)
# =========================
class CocoWrapper(data.Dataset):
    def __init__(self, img_root, ann_file, transforms=None):
        self.ds = tv.datasets.CocoDetection(img_root, ann_file)
        self.transforms = transforms
        cat_ids = sorted(self.ds.coco.getCatIds())
        self.cat_id_to_contig = {cid:i for i,cid in enumerate(cat_ids)}
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        img, anns = self.ds[idx]
        w,h = img.size
        boxes=[]; labels=[]
        for a in anns:
            if a.get("iscrowd",0)==1: continue
            x,y,bw,bh = a["bbox"]
            if bw<=1 or bh<=1: continue
            boxes.append([x,y,x+bw,y+bh])
            labels.append(self.cat_id_to_contig.get(a["category_id"], 0))
        if len(boxes)==0:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = clamp_boxes(torch.tensor(boxes, dtype=torch.float32), w, h)
            labels = torch.tensor(labels, dtype=torch.int64)
        img = tv.transforms.functional.pil_to_tensor(img).float()/255.0
        if self.transforms is not None:
            img = self.transforms(img)
        return img, {"boxes": boxes, "labels": labels, "size": torch.tensor([h,w])}

# =========================
# YOLO .txt wrapper
# =========================
class YoloTxtWrapper(data.Dataset):
    """
    Expect:
      images_dir/  *.jpg|*.png
      labels_dir/  same-stem .txt => <cls> <cx> <cy> <w> <h>  (normalized 0..1)
    """
    def __init__(self, images_dir, labels_dir, transforms=None, num_classes=None):
        from PIL import Image
        self.img_root = Path(images_dir); self.labels_dir = Path(labels_dir)
        self.transforms = transforms; self.Image = Image
        self.num_classes = num_classes
        exts = {".jpg",".jpeg",".png",".bmp",".webp"}
        self.img_paths = sorted([p for p in self.img_root.iterdir() if p.suffix.lower() in exts])

    def __len__(self): return len(self.img_paths)

    def _yolo_to_xyxy(self, w, h, cx, cy, bw, bh):
        x1 = (cx - bw/2)*w; y1 = (cy - bh/2)*h
        x2 = (cx + bw/2)*w; y2 = (cy + bh/2)*h
        return [x1, y1, x2, y2]

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = self.Image.open(img_path).convert("RGB")
        w,h = img.size
        txt_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes=[]; labels=[]
        if txt_path.exists():
            for line in open(txt_path,encoding="utf-8"):
                parts=line.strip().split()
                if len(parts)!=5: continue
                c,cx,cy,bw,bh = map(float, parts); c=int(c)
                if (self.num_classes is not None) and (c<0 or c>=self.num_classes):
                    continue
                boxes.append(self._yolo_to_xyxy(w,h,cx,cy,bw,bh)); labels.append(c)
        if len(boxes)==0:
            boxes=torch.zeros((0,4),dtype=torch.float32); labels=torch.zeros((0,),dtype=torch.int64)
        else:
            boxes=clamp_boxes(torch.tensor(boxes,dtype=torch.float32), w, h)
            labels=torch.tensor(labels,dtype=torch.int64)
        img_t = tv.transforms.functional.to_tensor(img)
        if self.transforms is not None:
            img_t = self.transforms(img_t)
        return img_t, {"boxes": boxes, "labels": labels, "size": torch.tensor([h,w])}

# =========================
# Resize (Tensor)
# =========================
class ResizeShortSide:
    def __init__(self, short_side=800, max_size=1333):
        self.short=short_side; self.max=max_size
    def __call__(self, img: torch.Tensor):
        C,H,W = img.shape
        s = self.short / min(H,W)
        newH,newW = int(round(H*s)), int(round(W*s))
        if max(newH,newW) > self.max:
            s2 = self.max / max(newH,newW)
            newH,newW = int(round(newH*s2)), int(round(newW*s2))
        return tv.transforms.functional.resize(img, [newH,newW])

train_aug = tv.transforms.Compose([
    tv.transforms.RandomHorizontalFlip(p=0.5),
])

# =========================
# Fast mAP@0.5 (11-point)
# =========================
def compute_map50(preds: List[Dict], gts: List[Dict], iou_thr=0.5, num_classes=1):
    ap_per_class = {}; eps=1e-9
    for cls in range(num_classes):
        cls_preds=[]; cls_gts=[]
        for p,g in zip(preds,gts):
            pm = (p["labels"]==cls); gm = (g["labels"]==cls)
            if pm.sum()==0 and gm.sum()==0: continue
            cls_preds.append({"boxes":p["boxes"][pm], "scores":p["scores"][pm]})
            cls_gts.append({"boxes":g["boxes"][gm]})
        flat=[]
        for i,cp in enumerate(cls_preds):
            for b,s in zip(cp["boxes"], cp["scores"]): flat.append((i, float(s), b))
        flat.sort(key=lambda x: -x[1])
        tp=[]; fp=[]; total=sum(len(x["boxes"]) for x in cls_gts); matched=[set() for _ in cls_gts]
        for i,score,box in flat:
            g = cls_gts[i]["boxes"] if i<len(cls_gts) else torch.empty((0,4))
            if len(g)==0: fp.append(1); tp.append(0); continue
            ious = box_iou(box.unsqueeze(0), g).squeeze(0)
            j = int(torch.argmax(ious)); v = float(ious[j])
            if v>=iou_thr and j not in matched[i]:
                matched[i].add(j); tp.append(1); fp.append(0)
            else:
                fp.append(1); tp.append(0)
        if total==0: ap_per_class[cls]=0.0; continue
        tp=np.cumsum(np.array(tp)); fp=np.cumsum(np.array(fp))
        rec=tp/(total+eps); prec=tp/(tp+fp+eps)
        ap=0.0
        for r in np.linspace(0,1,11):
            ap += prec[rec>=r].max() if np.any(rec>=r) else 0.0
        ap_per_class[cls]=ap/11.0
    mAP = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    return {"AP50_per_class": ap_per_class, "mAP50": mAP}

# =========================
# Build RetinaNet (with COCO pretrained)
# =========================
def build_retinanet(num_classes=12, min_anchor=16, aspect_ratios=(0.5,1.0,2.0),
                    focal_gamma=2.0, focal_alpha=0.25):
    try:
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        model = tv.models.detection.retinanet_resnet50_fpn_v2(
            weights=weights, num_classes=num_classes
        )
    except Exception:
        model = tv.models.detection.retinanet_resnet50_fpn(
            weights="DEFAULT", num_classes=num_classes
        )
    # anchors
    sizes=[]; s=min_anchor
    for _ in range(5):
        sizes.append((s,)); s*=2
    if hasattr(model, "anchor_generator"):
        model.anchor_generator.sizes = tuple(sizes)
        model.anchor_generator.aspect_ratios = tuple([aspect_ratios]*5)
    # focal params (if exposed)
    ch = getattr(getattr(model,"head",None), "classification_head", None)
    if ch is not None:
        if hasattr(ch,"gamma"): ch.gamma = focal_gamma
        if hasattr(ch,"alpha"): ch.alpha = focal_alpha
    return model

# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, optimizer, device, epoch, grad_accum=1, lr_scheduler=None, log_interval=20):
    model.train(); loss_meter=0.0; t0=time.time()
    for it,(images,targets) in enumerate(loader,1):
        images=[img.to(device) for img in images]
        targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values()) / grad_accum
        loss.backward()
        if it % grad_accum == 0:
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if lr_scheduler is not None: lr_scheduler.step()
        loss_meter += loss.item()*grad_accum
        if it % log_interval==0:
            print(f"[epoch {epoch} iter {it}/{len(loader)}] loss={loss_meter/it:.4f}")
    return loss_meter/max(1,len(loader)), time.time()-t0

@torch.no_grad()
def evaluate_map50(model, loader, device, num_classes):
    model.eval(); all_preds=[]; all_gts=[]; t0=time.time()
    for images, targets in loader:
        images=[img.to(device) for img in images]
        outputs=model(images)
        for o,t in zip(outputs,targets):
            all_preds.append({"boxes":o["boxes"].detach().cpu(),
                              "scores":o["scores"].detach().cpu(),
                              "labels":o["labels"].detach().cpu()})
            all_gts.append({"boxes":t["boxes"], "labels":t["labels"]})
    metrics = compute_map50(all_preds, all_gts, iou_thr=0.5, num_classes=num_classes)
    return metrics, time.time()-t0

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    # dataset mode
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--yolo-txt', action='store_true')
    # yolo: split paths
    parser.add_argument('--train-images', type=str, default='')
    parser.add_argument('--train-labels', type=str, default='')
    parser.add_argument('--val-images',   type=str, default='')
    parser.add_argument('--val-labels',   type=str, default='')
    # yolo: fallback (same dir for smoke)
    parser.add_argument('--data-root', type=str, default='')
    parser.add_argument('--labels-dir', type=str, default='')
    # coco paths
    parser.add_argument('--train-json', type=str, default='')
    parser.add_argument('--val-json', type=str, default='')

    # model / train
    parser.add_argument('--num-classes', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--short-side', type=int, default=800)
    parser.add_argument('--max-size', type=int, default=1333)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--min-anchor', type=int, default=16)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--focal-alpha', type=float, default=0.25)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out-dir', type=str, default='outputs_retinanet')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--grad-accum', type=int, default=1)

    # eval-only
    parser.add_argument('--eval-only', action='store_true', help='only evaluate on val loader and exit')
    parser.add_argument('--ckpt', type=str, default='', help='checkpoint for --eval-only')

    args = parser.parse_args()
    set_seed(42)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    use_pin_memory = (device.type=='cuda')

    # datasets
    if args.synthetic:
        train_ds = SyntheticDetectionDataset(n=64, img_size=512, num_classes=args.num_classes)
        val_ds   = SyntheticDetectionDataset(n=32, img_size=512, num_classes=args.num_classes)
        train_ds = WrapTransformDataset(train_ds, tv.transforms.Compose([train_aug]))
        val_ds   = WrapTransformDataset(val_ds,   tv.transforms.Compose([]))
    else:
        resize = ResizeShortSide(args.short_side, args.max_size)
        if args.yolo_txt:
            if args.train_images and args.train_labels and args.val_images and args.val_labels:
                train_ds = YoloTxtWrapper(args.train_images, args.train_labels,
                                          transforms=tv.transforms.Compose([resize, train_aug]),
                                          num_classes=args.num_classes)
                val_ds   = YoloTxtWrapper(args.val_images, args.val_labels,
                                          transforms=tv.transforms.Compose([resize]),
                                          num_classes=args.num_classes)
            else:
                assert args.data_root and args.labels_dir, \
                    "YOLO mode: set --train-images/--train-labels/--val-images/--val-labels, or fallback to --data-root & --labels-dir."
                train_ds = YoloTxtWrapper(args.data_root, args.labels_dir,
                                          transforms=tv.transforms.Compose([resize, train_aug]),
                                          num_classes=args.num_classes)
                val_ds   = YoloTxtWrapper(args.data_root, args.labels_dir,
                                          transforms=tv.transforms.Compose([resize]),
                                          num_classes=args.num_classes)
        else:
            assert args.data_root and args.train_json and args.val_json, \
                "COCO mode: set --data-root --train-json --val-json."
            train_ds = CocoWrapper(args.data_root, args.train_json,
                                   transforms=tv.transforms.Compose([resize, train_aug]))
            val_ds   = CocoWrapper(args.data_root, args.val_json,
                                   transforms=tv.transforms.Compose([resize]))

    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.workers, collate_fn=collate_fn, pin_memory=use_pin_memory)
    val_loader   = data.DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.workers, collate_fn=collate_fn, pin_memory=use_pin_memory)

    # model
    model = build_retinanet(num_classes=args.num_classes,
                            min_anchor=args.min_anchor,
                            aspect_ratios=(0.5,1.0,2.0),
                            focal_gamma=args.focal_gamma,
                            focal_alpha=args.focal_alpha).to(device)

    # eval-only
    if args.eval_only:
        assert args.ckpt and os.path.isfile(args.ckpt), "Provide --ckpt for --eval-only"
        ckpt = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)
        metrics, eval_time = evaluate_map50(model, val_loader, device, num_classes=args.num_classes)
        print(f"[EVAL-ONLY] mAP50={metrics['mAP50']:.4f}  (eval {eval_time:.1f}s)")
        return

    # optim & sched
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    milestones = [int(args.epochs*0.66), int(args.epochs*0.86)]
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    start_epoch=0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_sched.load_state_dict(ckpt['lr_sched'])
        start_epoch = ckpt['epoch']+1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # train loop
    best_map=-1.0; best_path=None
    for epoch in range(start_epoch, args.epochs):
        loss_avg, train_time = train_one_epoch(model, train_loader, optimizer, device, epoch, args.grad_accum, None)
        lr_sched.step()
        metrics, eval_time = evaluate_map50(model, val_loader, device, num_classes=args.num_classes)
        print(f"[epoch {epoch}] loss={loss_avg:.4f}  mAP50={metrics['mAP50']:.4f}  (train {train_time:.1f}s, eval {eval_time:.1f}s)")
        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "lr_sched": lr_sched.state_dict(), "args": vars(args), "mAP50": metrics["mAP50"]}
        path = os.path.join(args.out_dir, f"ckpt_epoch{epoch}_map50_{metrics['mAP50']:.4f}.pth")
        torch.save(ckpt, path)
        if metrics["mAP50"] > best_map:
            best_map = metrics["mAP50"]; best_path = os.path.join(args.out_dir, "best.pth")
            torch.save(ckpt, best_path); print(f"** Saved best to {best_path}")
    print(f"Training done. Best mAP50={best_map:.4f}, best ckpt={best_path}")

if __name__ == "__main__":
    main()
