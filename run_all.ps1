# run_all.ps1
$env:CUDA_VISIBLE_DEVICES="0"

python retina_yaml_train.py --yaml data.yaml `
  --epochs 12 --batch-size 2 --workers 0 `
  --short-side 1024 --max-size 1536 --min-anchor 8 `
  2>&1 | tee outputs_retinanet\train.log

python retina_yaml_eval.py --yaml data.yaml --ckpt outputs_retinanet\best.pth --split val
python retina_yaml_eval.py --yaml data.yaml --ckpt outputs_retinanet\best.pth --split test
