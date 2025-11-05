project_root/
│
├─ data.yaml                      # 你的数据路径 & 类别定义
│
├─ retinanet_train.py            # 主训练脚本（已支持 YOLO txt）
├─ retina_yaml_train.py          # 读取 data.yaml 开训练
├─ retina_yaml_eval.py           # 读取 data.yaml 做验证/测试
├─ infer_yolo_txt.py             # (test无标注时) 推理导出预测
│
├─ outputs_retinanet/            # 自动生成 (权重/日志/ckpt)
│    ├─ best.pth
│    ├─ train_*.log
│    └─ …
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
