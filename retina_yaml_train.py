# retina_yaml_train.py
import argparse, os, sys, subprocess, yaml
from pathlib import Path

def infer_labels_dir(images_dir: str) -> str:
    p = Path(images_dir)
    return str(p.parent / "labels") if p.name.lower()=="images" else str(p.parent / "labels")

def must_exist(path: str, what: str):
    if not Path(path).exists():
        print(f"[ERROR] {what} not found: {path}"); sys.exit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--script", default="retinanet_train.py")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--short-side", type=int, default=1024)
    ap.add_argument("--max-size",   type=int, default=1536)
    ap.add_argument("--min-anchor", type=int, default=8)
    ap.add_argument("--device", default=None)
    args, extra = ap.parse_known_args()

    cfg = yaml.safe_load(open(args.yaml, "r", encoding="utf-8"))
    num_classes = len(cfg["names"]) if "names" in cfg else int(cfg.get("nc", 1))
    train_images = cfg.get("train"); val_images = cfg.get("val")
    if not train_images or not val_images:
        print("[ERROR] data.yaml must contain 'train' and 'val'"); sys.exit(1)
    train_labels = infer_labels_dir(train_images); val_labels = infer_labels_dir(val_images)
    for p,w in [(train_images,"train images"),(train_labels,"train labels"),
                (val_images,"val images"),(val_labels,"val labels")]:
        must_exist(p,w)

    cmd = [sys.executable, args.script,
           "--yolo-txt",
           "--train-images", train_images, "--train-labels", train_labels,
           "--val-images",   val_images,   "--val-labels",   val_labels,
           "--num-classes", str(num_classes),
           "--epochs", str(args.epochs),
           "--batch-size", str(args.batch_size),
           "--workers", str(args.workers),
           "--short-side", str(args.short_side),
           "--max-size", str(args.max_size),
           "--min-anchor", str(args.min_anchor)]
    if args.device: cmd += ["--device", args.device]
    cmd += extra
    print("[RUN]", " ".join(cmd))
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
