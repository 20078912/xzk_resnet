# retina_yaml_eval.py
import argparse, os, sys, subprocess, yaml
from pathlib import Path

def infer_labels_dir(images_dir: str) -> str:
    p = Path(images_dir)
    return str(p.parent / "labels") if p.name.lower()=="images" else str(p.parent / "labels")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--script", default="retinanet_train.py")
    ap.add_argument("--split", choices=["val","test"], default="val")
    ap.add_argument("--num-classes", type=int, default=None)
    ap.add_argument("--short-side", type=int, default=1024)
    ap.add_argument("--max-size",   type=int, default=1536)
    args, extra = ap.parse_known_args()

    cfg = yaml.safe_load(open(args.yaml,"r",encoding="utf-8"))
    num_classes = args.num_classes if args.num_classes is not None else (len(cfg["names"]) if "names" in cfg else int(cfg.get("nc",1)))
    images = cfg.get(args.split)
    if not images: print(f"[ERROR] missing '{args.split}' in yaml"); sys.exit(1)
    labels = infer_labels_dir(images)
    if not Path(labels).exists():
        print(f"[ERROR] '{args.split}' has no labels dir -> cannot eval mAP. Found images={images}, expected labels={labels}")
        sys.exit(1)

    cmd = [sys.executable, args.script,
           "--yolo-txt",
           "--val-images", images, "--val-labels", labels,
           "--num-classes", str(num_classes),
           "--eval-only", "--ckpt", args.ckpt,
           "--short-side", str(args.short_side),
           "--max-size",   str(args.max_size)]
    cmd += extra
    print("[EVAL]", " ".join(cmd))
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
