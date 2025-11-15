# COMP9517 Computer Vision Group Project (2025 T3)

```
COMP9517_group_project/
│
├─ data.yaml                        # 数据集配置文件（见下方示例）
│
├─ retinanet_train.py               # 主训练脚本（当前使用的即可）
├─ retinanet_test.py                # 测试脚本（当前使用的即可）
├─ visualize_retinanet.py           # 可视化脚本（生成 success/failure 图）
│
├─ outputs_retinanet/               # 模型输出文件夹
│    ├─ best.pth
│    ├─ ckpt_epochX.pth
│    ├─ results_summary.csv
│    └─ ...
│
└─ dataset/
     ├─ train/
     │   ├─ images/
     │   └─ labels/
     ├─ valid/
     │   ├─ images/
     │   └─ labels/
     └─ test/
         ├─ images/
         └─ labels/

```   
operation :

python retinanet_train_final.py --train-images dataset/train/images --train-labels dataset/train/labels --val-images dataset/valid/images --val-labels dataset/valid/labels --num-classes 12 --epochs 20 --batch-size 3 --workers 8 --lr 0.002 --amp

python test_retinanet_final.py --ckpt outputs_retinanet_final/best.pth --test-images dataset/test/images --test-labels dataset/test/labels --num-classes 12


python visualize_retinanet_final.py

## 🪲 Insect Detection & Classification in Agriculture  
**Models:** RetinaNet • Faster R‑CNN • YOLO (or other CV methods your group chooses)

This repository contains our code submission for the COMP9517 Group Project (Term 3, 2025).  
The task is to detect and classify agricultural pest insects from the **AgroPest‑12** dataset.

---

## 📂 Dataset

**Dataset:** AgroPest‑12 (Kaggle)  
**Classes:** 12 agricultural insect categories  
**Images:** 11,502 train / 1,095 val / 546 test  
**Labels:** Bounding boxes + class labels  

Dataset link:  
https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset

> ⚠️ Dataset is **not included in this repo** due to size & submission rules.  
Please download manually and update local paths as needed.

---

## 🎯 Project Objectives

- Detect & classify insects in natural agricultural environments  
- Develop **at least 2 full detection pipelines** (detector + classifier)  
- Compare traditional, ML, and/or deep learning approaches  
- Evaluate robustness, speed, accuracy, and sensitivity to imbalance  
- Follow COMP9517 academic & coding guidelines  

---

## 🧠 Methods Overview

| Method | Detector | Notes |
|---|---|---|
| **Method 1** | e.g., Faster R‑CNN | Two‑stage baseline |
| **Method 2** | e.g., RetinaNet | One‑stage baseline |
| **Optional** | YOLO / SSD / Vision Transformer | For improvements & comparison |
| **Optional** | Classical + feature descriptors (SIFT/HOG + SVM) | For bonus diversity |

> Models and approaches will be updated as the project progresses.

---

## ⚙️ Environment & Dependencies

```
Python >= 3.9
PyTorch >= 1.12
torchvision >= 0.13
CUDA (optional but recommended)
```

Install dependencies (if requirements.txt is provided later):

```
pip install -r requirements.txt
```

---

## 🚀 Training Example

Example (custom parameters inside script):

```bash
python retinanet_train.py --epochs 50 --batch-size 8
```

> Replace with your script if name changes.

---

## 📦 Files Included

| File | Description |
|---|---|
| `retinanet_train.py` | RetinaNet training script |
| `data.yaml` | Dataset config file |
| `.gitignore` | Prevents dataset & weights from being committed |

**Not included** (per assignment rules):  
❌ Dataset  
❌ Trained weights  
❌ Output visualizations  

---

## 📊 Evaluation Metrics

- **mAP** (mean average precision) — detection performance  
- **Precision / Recall / F1** — classification  
- **AUC**
- **Inference & training time** comparisons  

---

## 🎥 Video + 📄 Report

Deliverables include:

- **10‑minute video presentation** (with live demo segment)
- **IEEE‑format report (max 10 pages)**

---

## 👥 Group Members

| Name | zID |
|---|---|
| Member 1 | z5xxxxx |
| Member 2 | z5xxxxx |
| Member 3 | z5xxxxx |
| Member 4 | z5xxxxx |
| Member 5 | z5xxxxx |

*(To be updated)*

---

## 📎 Acknowledgements

- UNSW COMP9517 Teaching Team  
- PyTorch / Torchvision  
- Kaggle dataset authors  

---

## 📜 License

For academic use only.  
COMP9517 submission – redistribution prohibited.

