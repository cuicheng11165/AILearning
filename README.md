# AILearning

AI 与深度学习核心概念笔记及可运行代码示例仓库。

## 目录结构

```text
AILearning/
├── README.md                                          # 项目总览与学习导航
├── docs/                                              # 理论与概念笔记
│   ├── 01-fundamentals/                               # 神经网络基础与核心理论
│   │   ├── 01-neural-network.md                       # 神经网络介绍
│   │   ├── 02-universal-approximation-theorem.md      # 万能近似定理介绍
│   │   └── 03-activation-functions.md                 # 激活函数（ReLU / Sigmoid / GELU 等）介绍
│   ├── 02-architectures/                              # 经典网络架构
│   │   ├── 01-mlp-and-fnn.md                          # 前馈神经网络（MLP / FNN）介绍
│   │   └── 02-cnn.md                                  # 卷积神经网络（CNN）介绍
│   └── assets/                                        # 文档配图与示意图（SVG）
│       ├── activation-functions.svg                   # Sigmoid / ReLU / GELU 函数与导数对比图
│       └── activation-functions-more.svg              # 其他常见激活函数（Tanh / SiLU / Mish 等）对比图
└── demos/                                             # 可运行代码示例（按主题分类）
    └── mlp/
        └── house_price_prediction.py                  # MLP 房价预测（纯 Python 零依赖 Demo）
```

---

## 学习笔记目录

### 01. 基础与理论（Fundamentals）

1. [神经网络介绍](./docs/01-fundamentals/01-neural-network.md) — 神经元结构、前向/反向传播、常见架构总览与优缺点。
2. [万能近似定理介绍](./docs/01-fundamentals/02-universal-approximation-theorem.md) — 为什么 MLP 理论上能逼近任意连续函数，以及为什么仍需要“深度”学习。
3. [激活函数（Activation Functions）介绍](./docs/01-fundamentals/03-activation-functions.md) — `Sigmoid`、`ReLU`、`GELU` 及常见变体的原理、对比与工程选型。

### 02. 经典网络架构（Architectures）

1. [前馈神经网络（MLP / FNN）介绍](./docs/02-architectures/01-mlp-and-fnn.md) — 全连接网络原理、表格数据建模以及作为基础积木在现代大模型中的角色。
2. [卷积神经网络（CNN）介绍](./docs/02-architectures/02-cnn.md) — 局部感受野与权重共享、卷积/池化组件、层级特征提取、经典架构演进及与 ViT 的对比。

---

## 代码示例（Demos）

代码按主题存放在 `demos/` 目录下，均可在仓库根目录直接运行：

| 主题 | 文件 | 说明 | 对应笔记 | 运行方式 |
| :--- | :--- | :--- | :--- | :--- |
| **MLP** | [house_price_prediction.py](./demos/mlp/house_price_prediction.py) | 纯 Python（零第三方依赖）实现 `4 → 16(ReLU) → 1` 房价预测，完整演示标准化、前向传播、MSE 损失、反向传播与预测 | [MLP / FNN 介绍](./docs/02-architectures/01-mlp-and-fnn.md) | `python demos/mlp/house_price_prediction.py` |
