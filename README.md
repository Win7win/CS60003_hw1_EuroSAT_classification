# HW1: 从零构建三层 MLP 实现 EuroSAT 地表覆盖分类

手工实现三层全连接神经网络（MLP），仅使用 NumPy 完成前向传播与反向传播，在 EuroSAT 卫星图像数据集上进行 10 类地表覆盖分类。

**测试集准确率：66.0%**

模型权重下载：https://drive.google.com/drive/folders/1dysWrnmL5lKeMKtsPJdX26FBc1z0KQVK?usp=share_link

---

## 目录结构

```
├── data/                        # 数据集目录（内容不上传，见 data/README.md）
├── train/                       # 核心模块：数据加载、模型、训练、超参数搜索
│   ├── data_loader.py
│   ├── model.py
│   ├── trainer.py
│   ├── hyperparameter_search.py
│   └── train.py                 # 训练入口
├── weight_visualization/
│   └── visualize.py             # 权重可视化与空间模式分析
├── error_analysis/
│   └── error_analysis.py        # 错例分析与混淆矩阵
├── outputs/                     # 运行后自动生成（图表、权重）
├── requirements.txt
└── README.md
```

---

## 环境配置

```bash
conda create -n deeplearning python=3.10
conda activate deeplearning
pip install -r requirements.txt
```

---

## 数据集

按 `data/README.md` 的说明将 EuroSAT_RGB 文件夹放入 `data/` 目录下。

---

## 运行说明

所有脚本均从各自所在目录运行。

### 训练

```bash
cd train

# 默认配置直接训练（lr=0.01, hidden=512/256, relu, wd=1e-4, 60 epoch）
python train.py

# 先做网格搜索，再用最优配置全量训练
python train.py --search grid

# 先做随机搜索，再用最优配置全量训练
python train.py --search random

# 自定义超参数
python train.py --lr 0.005 --hidden_dim1 512 --hidden_dim2 256 --epochs 80

# 仅评估（跳过训练，加载已有权重）
python train.py --test --ckpt ../outputs/best_model.npz
```

训练完成后在 `outputs/` 下生成：
- `best_model.npz` — 验证集最优权重
- `norm_stats.npy` — 数据归一化统计量
- `training_curves.png` — Loss / Accuracy 曲线
- `history.json` — 完整训练历史

### 测试（加载权重，输出准确率 + 混淆矩阵）

```bash
cd train
# 数据在默认位置 ../data/EuroSAT_RGB
python test.py --ckpt ../outputs/best_model.npz

# 或指定数据路径
python test.py --ckpt ../outputs/best_model.npz --data_dir /path/to/EuroSAT_RGB
```

输出测试集准确率、完整混淆矩阵、各类别准确率。

### 权重可视化与空间模式分析

```bash
cd weight_visualization
python visualize.py --ckpt ../outputs/best_model.npz
```

生成：
- `outputs/weight_vis.png` — W1 全量列可视化（64 个神经元）
- `outputs/weight_class_analysis.png` — 各类别平均图像 + top 激活神经元
- `outputs/weight_stats.png` — 各类激活强度柱状图 & RGB 通道偏置

### 错例分析

```bash
cd error_analysis
python error_analysis.py --ckpt ../outputs/best_model.npz
```

生成：
- `outputs/confusion_matrix.png` — 混淆矩阵
- `outputs/error_analysis.png` — 随机抽取的错误分类样本

---

## 模型结构

```
输入 (12288) → 全连接 (512, ReLU) → 全连接 (256, ReLU) → 输出 (10, Softmax)
```

- 损失函数：Softmax Cross-Entropy + L2 正则化
- 优化器：Mini-batch SGD，学习率指数衰减（×0.95 / 5 epoch）
- 权重初始化：He 初始化（ReLU）

---

## 主要结果

| 指标 | 数值 |
|------|------|
| 测试集准确率 | 66.0% |
| 最优验证集准确率 | 66.4%（epoch 59） |
| 最佳单类准确率 | Forest 93.7% |
| 最低单类准确率 | Highway 41.9% |
