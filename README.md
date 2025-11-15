# RetinaNet Detection Pipeline — Summary README

This repository provides a complete RetinaNet-based object detection pipeline for datasets formatted in YOLO TXT annotations. It includes training, validation, testing, visualization, and metric reporting, producing structured logs, CSV files, and diagnostic figures.

---

## Project Structure

```
project/
│
├── retinanet_train_full_final.py      # Full-feature RetinaNet training script
├── retinanet_train_final.py           # Lightweight training variant
├── test_retinanet_final.py            # Test/evaluation script
├── visualize_retinanet_final.py       # Detection visualization script
│
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
├── outputs_retinanet_final/
│   ├── best.pth
│   ├── last.pth
│   ├── training_log.csv
│   ├── loss_curve.png
│   ├── map_curve.png
│   ├── pr_curve.png
│
└── outputs_retinanet_test/
    ├── class_results.csv
    ├── per_class_map.png

└── vis_results_final/
    ├── success/
    ├── failure/
    ├── summary_success.jpg
    ├── summary_failure.jpg
    └── results_summary.csv
```

---

## Dataset Format (YOLO TXT)

Each label file:

```
class center_x center_y width height
```

All values are normalized to 0–1.

---

## Installation

```
pip install torch torchvision numpy pillow pandas scikit-learn matplotlib
```

---

## Training

Run:

```
python retinanet_train_full_final.py   --train-images dataset/train/images   --train-labels dataset/train/labels   --val-images dataset/valid/images   --val-labels dataset/valid/labels   --num-classes 12   --epochs 20   --batch-size 3   --workers 8   --lr 0.002   --amp
```

### Training Outputs (auto-generated)

- best.pth  
- last.pth  
- training_log.csv  
- loss_curve.png  
- map_curve.png  
- pr_curve.png  

---

## Testing

Run:

```
python test_retinanet_final.py   --ckpt outputs_retinanet_final/best.pth   --test-images dataset/test/images   --test-labels dataset/test/labels   --num-classes 12
```

### Console Summary Includes

- mAP@50  
- mAP@50–95  
- Detection precision & recall  
- Classification precision, recall, F1  
- Classification accuracy  
- ROC-AUC  
- Evaluation time  

### Test Output Files

Located in `outputs_retinanet_test/`:

- class_results.csv  
- per_class_map.png  

---

## Visualization

Run:

```
python visualize_retinanet_final.py
```

### Visualization Outputs

Located in `vis_results_final/`:

- success/  
- failure/  
- summary_success.jpg  
- summary_failure.jpg  
- results_summary.csv  

---

## Summary

This repository provides a full RetinaNet workflow including training, evaluation, metric reporting, per-class analysis, and visualization. It supports YOLO TXT datasets and produces structured logs and figures suitable for research and experimentation.
