# VecFormer 入门指南

> 面向仅有 Python 基础的读者。从"项目做什么"到"代码怎么工作"，逐步建立完整认知。

---

## 目录

1. [项目是干什么的](#1-项目是干什么的)
2. [必备知识储备](#2-必备知识储备)
3. [项目整体结构](#3-项目整体结构)
4. [数据从哪来、长什么样](#4-数据从哪来长什么样)
5. [模型架构：从输入到输出](#5-模型架构从输入到输出)
6. [训练流程](#6-训练流程)
7. [代码阅读路线图](#7-代码阅读路线图)
8. [常见概念速查](#8-常见概念速查)
9. [推理流程详解](#9-推理流程详解)
10. [项目中的设计模式](#10-项目中的设计模式)
11. [完整前向传播追踪](#11-完整前向传播追踪)
12. [常见问题与解决](#12-常见问题与解决)

---

## 1. 项目是干什么的

### 任务背景

建筑师画 CAD 图时，图纸里有成千上万条线段（线段 = **图元/primitive**）。
这些线段组成门、窗、墙等建筑符号（**实例**），同时每条线都属于某个语义类别（如"门"、"窗"）。

**目标**：给定一张 CAD 图，自动识别出：
- 每条线段属于哪个语义类别（**语义分割**）
- 哪些线段属于同一个建筑符号实例（**实例分割**）
- 二者合并叫 **全景分割（Panoptic Segmentation）**

### 本项目的创新

传统方法把 CAD 图转成图片再处理（丢失了精确坐标）。VecFormer 直接把线段作为输入，保留了向量图形的精确几何信息。

```
传统方法：线段 → 栅格图像 → CNN → 结果（精度损失）
VecFormer：线段序列 → Transformer → 结果（保留精度）
```

---

## 2. 必备知识储备

按优先级排列，建议依次学习：

### 第一层：Python 进阶（1-2 周）

| 概念 | 为什么需要 | 在代码里的体现 |
|------|-----------|---------------|
| `@dataclass` 装饰器 | 定义数据结构 | `SVGData`, `VecData` |
| `typing` 模块（`list[int]`, `Optional`） | 类型注解 | 几乎所有文件 |
| `*args`, `**kwargs` | 灵活参数传递 | `VecFormerConfig.__init__` |
| 列表/字典推导式 | 数据处理 | `data/floorplancad/` |
| `with` 语句、上下文管理器 | 文件操作 | `preprocess.py` |

### 第二层：NumPy + PyTorch 基础（2-3 周）

**NumPy**（理解张量操作的基础）：
- 数组创建、索引、切片
- `reshape`, `transpose`, `concatenate`
- 广播机制（broadcasting）

**PyTorch**（项目核心框架）：

```python
# 必须掌握的 5 个核心概念：

# 1. Tensor（张量）= 多维数组
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape: (2, 2)

# 2. nn.Module（模块）= 神经网络的基本单元
class MyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 128)  # 全连接层

    def forward(self, x):          # 定义前向计算
        return self.linear(x)

# 3. 自动求导
y = model(x)
loss = criterion(y, labels)
loss.backward()    # 自动计算梯度

# 4. 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer.step()   # 更新参数

# 5. 设备管理
device = "cuda"    # GPU
x = x.to(device)  # 把数据移到 GPU
```

### 第三层：深度学习核心概念（3-4 周）

#### 3.1 神经网络基础

```
输入 → [线性层 + 激活函数] × N → 输出
```

- **线性层（Linear）**：`y = Wx + b`，学习特征变换
- **激活函数（GELU/ReLU）**：引入非线性
- **LayerNorm**：归一化，让训练更稳定
- **Dropout**：随机关闭部分神经元，防止过拟合

#### 3.2 注意力机制（Attention）——项目核心

这是理解整个项目最关键的概念。

**直觉理解**：你在读一个句子时，理解某个词的意思需要参考其他词。注意力机制就是让模型学会"哪些位置对当前位置重要"。

```
Query（我想要什么）× Key（我有什么）→ 注意力权重
注意力权重 × Value（实际内容）→ 输出
```

**代码位置**：`model/vecformer/modules/attention.py`

```python
# 项目中的变长自注意力（VarlenSelfAttention）
# 输入：N 条线段的特征，每条线段互相关注其他线段
# 输出：融合了上下文信息的特征

attn_output = flash_attn.flash_attn_varlen_qkvpacked_func(
    qkv_packed,      # Query, Key, Value 打包在一起
    cu_seqlens,      # 每张图有多少条线（可变长度）
    max_seqlen,
)
```

#### 3.3 Transformer 架构

VecFormer 的核心骨架，由注意力层堆叠而成：

```
输入特征
    ↓
[自注意力（Self-Attention）] ← 每条线段关注其他线段
    ↓
[加法 + 归一化（Add & Norm）]
    ↓
[前馈网络（FFN）] ← 独立处理每条线段
    ↓
[加法 + 归一化（Add & Norm）]
    ↓
输出特征（包含上下文信息）
```

**代码位置**：`model/vecformer/modules/transformer_block.py`

#### 3.4 编码器-解码器（Encoder-Decoder）

```
Encoder（编码器）：把输入压缩成高级特征表示
    输入：7维线段特征 → 输出：128维语义特征

Decoder（解码器）：把高级特征解码为目标输出
    输入：编码器输出 + 查询向量 → 输出：实例预测
```

### 第四层：计算机视觉中的分割任务（1-2 周）

#### 4.1 语义分割 vs 实例分割 vs 全景分割

```
语义分割：每个像素/元素 → 类别标签（门/窗/墙）
          [线1→门, 线2→门, 线3→窗, 线4→墙]

实例分割：区分同一类别的不同个体
          [线1→门#1, 线2→门#2, 线3→窗#1, 线4→墙#1]

全景分割：语义 + 实例的结合
          [线1→(门, 实例#1), 线2→(门, 实例#2), ...]
```

#### 4.2 匈牙利算法（Hungarian Algorithm）

训练时需要把"预测的实例"与"真实标注的实例"配对。用匈牙利算法找最优配对方案（二分图最优匹配），匹配后再计算损失。

**代码位置**：`model/vecformer/criterion/instance_criterion.py`

#### 4.3 评估指标：Panoptic Quality (PQ)

```
PQ = SQ × RQ

SQ（分割质量）= 匹配对的平均 IoU
RQ（识别质量）= TP / (TP + 0.5 × FP + 0.5 × FN)

IoU（交并比）= 预测 ∩ 真实 / 预测 ∪ 真实
```

### 第五层：点云与序列处理（按需学习）

- **点云**：用离散点集合表示空间结构。项目里的 PointTransformerV3 处理"点云"形式的 CAD 图
- **稀疏卷积（spconv）**：大部分空间为空时，只计算有值的区域
- **序列打包（packed sequence）**：多条长度不同的序列放入同一 batch 的技巧

---

## 3. 项目整体结构

```
VecFormer/
├── launch.py                   # ← 程序入口，从这里开始
├── configs/                    # 配置文件
│   ├── vecformer.yaml          #   训练超参数（学习率、batch size 等）
│   ├── model/vecformer.yaml    #   模型配置
│   └── data/floorplancad.yaml  #   数据集配置
├── scripts/                    # 运行脚本
│   ├── train.sh                #   训练
│   ├── test.sh                 #   测试
│   └── resume.sh               #   断点续训
├── model/                      # 模型代码
│   └── vecformer/
│       ├── modeling_vecformer.py       # 主模型（总指挥）
│       ├── configuration_vecformer.py  # 配置类
│       ├── modules/            #   可复用组件（注意力、FFN 等）
│       ├── vec_backbone/       #   特征提取骨干网络
│       ├── point_transformer_v3/  # PTv3 骨干（第三方改编）
│       ├── cad_decoder/        #   实例/语义解码器
│       ├── criterion/          #   损失函数
│       ├── evaluator/          #   评估指标
│       └── vecformer_trainer.py   # 训练器
├── data/                       # 数据集代码
│   └── floorplancad/
│       ├── floorplancad.py     #   数据集类
│       ├── dataclass_define.py #   数据结构定义
│       ├── preprocess.py       #   数据预处理脚本
│       └── augment_utils.py    #   数据增强
└── utils/                      # 工具函数
```

---

## 4. 数据从哪来、长什么样

### 4.1 原始数据：FloorPlanCAD 数据集

每个样本是一张建筑平面图的 JSON 文件，记录了所有线段的信息。

### 4.2 一条线段有哪些属性

```python
# 代码位置：data/floorplancad/dataclass_define.py
@dataclass
class SVGData:
    coords:        # 线段端点坐标 [[x1,y1,x2,y2], ...]
    colors:        # 颜色 RGB [[r,g,b], ...]
    widths:        # 线宽 [w1, w2, ...]
    primitive_ids: # 图元ID（同一个符号的线段有相同ID）
    layer_ids:     # CAD图层ID
    semantic_ids:  # 语义类别标签（门=0, 窗=1, ...）
    instance_ids:  # 实例ID（第1扇门=0, 第2扇门=1, ...）
```

### 4.3 模型实际接收的输入（Line 模式）

原始数据经过处理后，每条线段变成：

```python
# 坐标（4维）：线段两端点坐标
coords = [x1, y1, x2, y2]

# 特征（7维）：
feats = [
    length,          # 线段长度
    abs(dx),         # 水平跨度（绝对值）
    abs(dy),         # 垂直跨度（绝对值）
    center_x,        # 中心点 x
    center_y,        # 中心点 y
    red/255,         # 颜色 R（归一化）
    green/255,       # 颜色 G（归一化）
    # blue 省略，因为 r+g+b 中有一个冗余
]
```

### 4.4 批处理的关键技巧：变长序列打包

不同图纸的线段数量不同（有的200条，有的2000条）。不能像图像那样 pad 到相同大小（太浪费），而是把多张图的线段**直接拼接**，用 `cu_seqlens`（累计序列长度）记录边界：

```python
# 例：3 张图分别有 100, 300, 200 条线段
# 拼接后总共 600 条线，feats shape = (600, 7)

cu_seqlens = [0, 100, 400, 600]
#              ^   ^    ^    ^
#              图1起 图2起 图3起  结束

# 取第 i 张图的数据：
# feats[cu_seqlens[i] : cu_seqlens[i+1]]
```

---

## 5. 模型架构：从输入到输出

### 5.1 整体数据流

```
输入（一批CAD线段）
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  PointTransformerV3 骨干网络                          │
│                                                     │
│  原始特征(7维) → [稀疏卷积 + 序列化注意力] × 5层       │
│                     ↓ 下采样                         │
│  [32, 64, 128, 256, 512] 维特征图                    │
│                     ↓ 上采样解码                     │
│  多尺度特征 [64, 64, 128, 256] 维                    │
└─────────────────────────────────────────────────────┘
         │  骨干输出：每条线段的 64 维特征
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Layer Fusion（层融合，可选）                          │
│  把不同 CAD 图层的线段特征聚合，增强上下文              │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  CAD Decoder（解码器）                                │
│                                                     │
│  初始化 N 个"查询向量"（代表 N 个候选实例）             │
│                                                     │
│  重复 6 次：                                         │
│    1. [自注意力] 查询向量互相交流                      │
│    2. [交叉注意力] 查询向量从骨干特征中提取信息          │
│    3. [FFN] 进一步处理                               │
│    4. [预测头] 当前状态下预测实例掩码和类别             │
└─────────────────────────────────────────────────────┘
         │
         ▼
输出：
  - 语义预测：每条线属于哪个类别（600类→35类）
  - 实例预测：每条线属于哪个实例（600个查询×N条线的掩码矩阵）
```

### 5.2 骨干网络：PointTransformerV3

**代码位置**：`model/vecformer/point_transformer_v3/model.py`

关键设计——**序列化注意力（Serialized Attention）**：

直接对所有线段做全局注意力，计算量是 O(N²)，线段数多时太慢。PTv3 的做法：
1. **空间序列化**：把线段按 Z-order 曲线或 Hilbert 曲线排序，空间上相邻的线段排在一起
2. **局部窗口注意力**：每次只对相邻的 1024 条线段做注意力（patch_size=1024）
3. **多顺序**：用 4 种不同的排序方式（`z, z-trans, hilbert, hilbert-trans`），避免窗口边界的线段被"分开"

```python
# 代码位置：configuration_vecformer.py
order=("z", "z-trans", "hilbert", "hilbert-trans"),  # 4 种排序
enc_patch_size=(1024, 1024, 1024, 1024, 1024),        # 每次处理 1024 条线
```

### 5.3 CAD 解码器

**代码位置**：`model/vecformer/cad_decoder/cad_decoder.py`

解码器使用"查询向量"（query）机制，这是 DETR 系列模型的经典设计：

```
初始查询向量（随机初始化，可学习）
         │
         ▼  × 6层
[自注意力：查询向量互相交流，避免预测重复的实例]
         ↓
[交叉注意力：查询向量"询问"骨干特征，提取局部信息]
         ↓
[FFN：进一步处理]
         ↓
[预测头：预测这个查询对应的实例]
  ├── 类别预测：这是什么（门/窗/墙...）？
  ├── 掩码预测：哪些线段属于这个实例？
  └── 置信度分数：预测有多可靠？
```

### 5.4 损失函数

**代码位置**：`model/vecformer/criterion/`

训练时有两个损失：

```
总损失 = 实例损失 + 语义损失

实例损失（InstanceCriterion）：
  步骤1：用匈牙利算法把预测实例与真实实例配对
  步骤2：对配对的实例计算：
    - 类别损失（Cross Entropy）：预测类别与真实类别的差距
    - 掩码损失（BCE + Dice）：预测掩码与真实掩码的差距
    - 置信度损失：预测分数与 IoU 的差距

语义损失（SemanticCriterion）：
  - 对每条线段预测类别（Cross Entropy）
```

---

## 6. 训练流程

### 6.1 一次训练迭代

```python
# 伪代码，简化自 launch.py + vecformer_trainer.py

for batch in dataloader:
    # 1. 前向传播
    outputs = model(batch)

    # 2. 计算损失
    loss = outputs.loss   # 已在模型内部计算

    # 3. 反向传播（自动计算梯度）
    loss.backward()

    # 4. 更新参数
    optimizer.step()
    optimizer.zero_grad()

    # 5. 定期评估（每 2175 步）
    if step % eval_steps == 0:
        metrics = evaluate(model, val_dataset)
        print(f"PQ: {metrics['PQ']:.4f}")
```

### 6.2 关键训练配置

```yaml
# configs/vecformer.yaml
num_train_epochs: 500              # 训练 500 轮
per_device_train_batch_size: 2     # 每块 GPU 每次处理 2 张图
learning_rate: 0.0001              # 初始学习率
lr_scheduler_type: warmup_stable_decay  # 学习率先热身，再稳定，再衰减
metric_for_best_model: PQ          # 用全景质量选最优模型
```

### 6.3 运行训练

```bash
# 单机 8 GPU 训练
bash scripts/train.sh

# 单 GPU 测试（修改 NPROC_PER_NODE）
NPROC_PER_NODE=1 bash scripts/train.sh

# 只做评估
bash scripts/test.sh
```

---

## 7. 代码阅读路线图

**建议按以下顺序阅读代码**，从外到内，从简单到复杂：

### 阶段一：理解数据（2-3 天）

```
1. data/floorplancad/dataclass_define.py
   → 了解 SVGData 和 VecData 的数据结构

2. data/floorplancad/floorplancad.py
   → 了解数据集如何加载和预处理
   → 重点看 __getitem__() 和 collate_fn()

3. data/floorplancad/augment_utils.py
   → 了解数据增强（翻转、旋转、缩放）
```

### 阶段二：理解配置（1 天）

```
4. model/vecformer/configuration_vecformer.py
   → 了解所有超参数的含义
   → 特别关注 backbone_config 和 cad_decoder_config
```

### 阶段三：理解基础组件（3-5 天）

```
5. model/vecformer/modules/attention.py
   → VarlenSelfAttention（变长自注意力）
   → 理解 cu_seqlens 的作用

6. model/vecformer/modules/transformer_block.py
   → TransformerBlock（注意力 + FFN 的组合）

7. model/vecformer/modules/group_feat_fusion.py
   → GroupFeatFusion（按图元/图层聚合特征）
```

### 阶段四：理解骨干网络（5-7 天）

```
8. model/vecformer/point_transformer_v3/model.py
   → SerializedAttention（序列化注意力）
   → PointTransformerV3（整个骨干网络）
   → 重点：enable_flash 分支和 upcast 分支

9. model/vecformer/vec_backbone/vec_backbone.py
   → VecBackbone（向量图形专用骨干）
   → 理解编码器-解码器结构
```

### 阶段五：理解解码与损失（5-7 天）

```
10. model/vecformer/cad_decoder/cad_decoder.py
    → CADDecoder（实例解码器）
    → 理解查询向量机制

11. model/vecformer/criterion/instance_criterion.py
    → 匈牙利匹配 + 实例损失

12. model/vecformer/criterion/semantic_criterion.py
    → 语义分割损失

13. model/vecformer/evaluator/evaluator.py
    → PQ 指标计算
```

### 阶段六：理解主模型和训练（3 天）

```
14. model/vecformer/modeling_vecformer.py
    → VecFormer（主模型，整合所有组件）
    → forward()：训练时的完整流程
    → predict()：推理时的完整流程

15. launch.py
    → 程序入口，理解整体训练流程

16. model/vecformer/vecformer_trainer.py
    → 自定义训练器（基于 HuggingFace Trainer）
```

---

## 8. 常见概念速查

| 术语 | 含义 | 代码位置 |
|------|------|---------|
| `cu_seqlens` | 累计序列长度，用于标记 batch 中每张图的边界 | 几乎所有前向函数 |
| `primitive` / `prim` | 图元，即一条线段 | `dataclass_define.py` |
| `VecData` | 模型输入的数据结构 | `dataclass_define.py` |
| `enable_flash` | 是否启用 Flash Attention（RTX 3090+ 推荐开启） | `configuration_vecformer.py:42` |
| `SerializedAttention` | 序列化局部注意力（PTv3 的核心） | `point_transformer_v3/model.py` |
| `CADDecoder` | 实例+语义解码头 | `cad_decoder/cad_decoder.py` |
| `PQ` | 全景质量（Panoptic Quality），主要评估指标 | `evaluator/evaluator.py` |
| `IoU` | 交并比，衡量预测与真实的重叠程度 | `evaluator/evaluator.py` |
| `Hungarian` | 匈牙利算法，用于训练时预测与真值的最优配对 | `instance_criterion.py` |
| `upcast_attention` | 注意力计算升级为 float32，提高数值稳定性 | `point_transformer_v3/model.py` |
| `iter_pred` | 解码器每层都输出预测并计算损失，加速训练收敛 | `configuration_vecformer.py:63` |
| `spconv` | 稀疏卷积库，处理大量空白区域效率高 | `point_transformer_v3/model.py` |
| `torchrun` | PyTorch 多 GPU 分布式训练启动工具 | `scripts/train.sh` |
| `AdamW` | 带权重衰减的 Adam 优化器，深度学习标配 | `configs/vecformer.yaml` |
| `warmup` | 训练初期用小学习率逐渐升到目标值，防止早期不稳定 | `configs/vecformer.yaml` |

---

## 9. 推理流程详解

训练完成后，模型如何把输出变成最终的"哪条线属于哪个实例/类别"？

### 9.1 原始输出

CAD 解码器的最后一块（block 6）产生：

```
pred_sem_labels  : (N, 35)     每条线段属于各类别的分数（logits）
pred_inst_masks  : (Q, N)      Q 个查询向量对 N 条线段的掩码分数
pred_inst_labels : (Q, 35)     Q 个查询向量各类别的分数
pred_inst_scores : (Q, 1)      Q 个查询向量的置信度

其中：N = 该图总线段数，Q = 查询向量数（训练时动态确定，推理时 = N）
```

### 9.2 语义预测（predict_semantic）

```python
# 代码位置：modeling_vecformer.py → predict_semantic()

# 1. Softmax：把 logits 变成概率
probs = softmax(pred_sem_labels, dim=-1)   # (N, 35)

# 2. 取最大值：每条线段的预测类别和置信度
pred_class = argmax(probs, dim=-1)   # (N,)  → 类别 ID
pred_score = max(probs, dim=-1)      # (N,)  → 最大概率
```

### 9.3 实例预测（predict_instance）

```python
# 代码位置：modeling_vecformer.py → predict_instance()

# 1. 过滤低置信度查询（pred_score_thr = 0.5）
keep = pred_inst_scores > 0.5

# 2. 激活掩码（sigmoid 变成 0~1 的概率）
masks = sigmoid(pred_inst_masks[keep])   # (K, N)

# 3. 过滤太小的实例（太少的线段）
valid_masks = masks > mask_logit_thr   # (K, N) → 布尔掩码

# 4. 对象归一化（可选）：用实例内部的线段长度加权分数
#    避免把一条很长线的"局部匹配"误判为完整实例

# 5. 取 Top-K（num_topk_preds = 600）
#    按分数排序，只保留最好的 600 个预测
```

### 9.4 全景融合（predict_panoptic）

语义预测和实例预测可能冲突（同一条线既被预测为"门实例#1"也被预测为"门"语义），需要融合：

```
对每条线段，投票决定其归属：
1. 如果它在某个置信度足够高的实例掩码中 → 归属该实例
2. 如果它属于 Stuff 类别（背景类：30-34）→ 只做语义不做实例
3. 冲突时，得分高的实例优先

最终每条线得到：(语义类别, 实例ID) 二元组
```

### 9.5 Vector NMS（向量非极大值抑制）

```python
# 代码位置：model/vecformer/modules/__init__.py → vector_nms()

# 功能：去除重复的实例预测
# 原理：两个预测掩码的 IoU > 阈值时，保留置信度高的，去除低的
# 专为向量图形设计，使用线段级别的 IoU 计算
```

---

## 10. 项目中的设计模式

理解这些设计模式有助于快速读懂整个代码库。

### 10.1 注册表模式（Registry Pattern）

项目用注册表把"模型名称字符串"映射到"实际的模型类"，实现了配置与代码的解耦：

```python
# 代码位置：model/__init__.py

# 注册
@register_model("vecformer")
def build(model_args: dict):
    return VecFormer(VecFormerConfig(**model_args)), VecFormerTrainer

# 使用（在 launch.py 中）
model, trainer_class = build_model("vecformer", model_args)
# ↑ 只需要传字符串名称，不需要 import 具体类
```

好处：要添加新模型，只需新建文件 + 一个 `@register_model("my_model")`，launch.py 无需任何修改。

数据集同理，使用 `@register_dataset("floorplancad")`（`data/__init__.py`）。

### 10.2 HuggingFace Trainer 模式

项目基于 HuggingFace `transformers` 库的 `Trainer` 类，这个类封装了训练循环、评估、存档等大量样板代码：

```python
# 代码位置：model/vecformer/vecformer_trainer.py

class VecFormerTrainer(Trainer):
    # 只需要重写需要定制的部分

    def compute_loss(self, model, inputs, ...):
        # 自定义损失计算

    def prediction_step(self, model, inputs, ...):
        # 自定义推理步骤（返回预测结果供评估）
```

这意味着：分布式训练、混合精度、梯度累积、进度条等功能都由 HuggingFace 自动处理。

### 10.3 PretrainedConfig 模式

`VecFormerConfig` 继承自 HuggingFace 的 `PretrainedConfig`，好处：

```python
# 保存配置
config.save_pretrained("outputs/my_model/")
# → 自动生成 outputs/my_model/config.json

# 加载配置
config = VecFormerConfig.from_pretrained("outputs/my_model/")
# → 从 JSON 恢复完整配置
```

### 10.4 PointModule 模式

`PointTransformerV3` 中使用 `PointModule` / `PointSequential` 代替标准的 `nn.Module` / `nn.Sequential`，因为点云数据不能简单地用张量传递（每个点有坐标、特征、批次信息等），需要用字典 `point` 打包：

```python
# 标准 PyTorch 写法
x = layer1(x)
x = layer2(x)

# PointModule 写法
point = layer1(point)   # point 是一个包含 feat/coord/offset 等的字典
point = layer2(point)
```

---

## 11. 完整前向传播追踪

用具体数字追踪一次推理，建立对张量维度变化的直观认识。

**假设**：一批 2 张图，分别有 300 和 500 条线段，总计 800 条。

```
输入
  coords      : (800, 4)   每条线的 [x1, y1, x2, y2]
  feats       : (800, 7)   每条线的特征
  cu_seqlens  : [0, 300, 800]   批次边界

── PointTransformerV3 骨干网络 ──────────────────────────────

  步骤1：体素化 & 稀疏卷积
    feats: (800, 7) → 投影 → (800, 32)   [enc_channels[0] = 32]

  步骤2：序列化（Hilbert 曲线排序）
    对 800 条线段重新排序，空间相邻的聚在一起

  步骤3：编码器层1（深度=2, 通道=32）
    每次做局部注意力（patch_size=1024，这里 800<1024，所以全局）
    输出：(800, 32)

  步骤4：编码器层2（下采样 stride=2, 通道=64）
    下采样：800 → 400 条关键线段
    输出：(400, 64)

  步骤5：编码器层3 → (200, 128)
  步骤6：编码器层4 → (100, 256)（深度=6）
  步骤7：编码器层5 → (50, 512)

  步骤8：解码器（逐步上采样恢复分辨率）
    (50, 512) + skip(100, 256) → (100, 256)
    (100, 256) + skip(200, 128) → (200, 128)
    (200, 128) + skip(400, 64)  → (400, 64)
    (400, 64)  + skip(800, 32)  → (800, 64)   [dec_channels[0]=64]

  骨干输出：(800, 64)   每条线段有 64 维语义特征

── Layer Fusion（图层融合）──────────────────────────────────

  layer_ids 标记每条线属于哪个 CAD 图层
  GroupFeatFusion：同一图层的线段特征聚合求均值，再拼接回来
  输出：(800, 64)   形状不变，但融入了图层级上下文

── CAD Decoder ──────────────────────────────────────────────

  查询向量初始化（推理时 Q = N = 800）
    queries: (800, 64) ← 来自骨干特征的采样

  解码器 Block × 6：
    自注意力：queries (800, 64) 内部交流  → (800, 64)
    交叉注意力：queries 查询 backbone feats → (800, 64)
    FFN       : (800, 64) → (800, 64)
    预测头    : → 本轮预测结果

  最终输出（来自第6块）：
    pred_sem_labels  : (800, 35)    语义类别分数
    pred_inst_masks  : (800, 800)   实例掩码矩阵
    pred_inst_labels : (800, 35)    实例类别分数
    pred_inst_scores : (800, 1)     实例置信度

── 后处理 ───────────────────────────────────────────────────

  过滤 + NMS + 投票后，最终输出：
    每条线段 → (语义类别 ID, 实例 ID)
```

---

## 12. 常见问题与解决

### Q1：CUDA out of memory（显存不足）

```bash
# 原因：batch size 太大，或图纸线段数太多
# 解决：减小 batch size
per_device_train_batch_size: 1   # configs/vecformer.yaml

# 或者减小 patch_size（减少单次注意力计算量）
enc_patch_size: (512, 512, 512, 512, 512)   # 默认是 1024
```

### Q2：flash_attn 安装失败 / RuntimeError

```bash
# 在 RTX 2080 Ti（sm_75）上，指定架构编译：
CUDA_ARCH=75 MAX_JOBS=8 python setup.py install

# 如果仍有问题，直接禁用 Flash Attention：
# 在 configuration_vecformer.py 中：
backbone_config = dict(
    ...
    enable_flash=False,       # 关闭
    upcast_attention=True,    # 开启以保证数值稳定
    upcast_softmax=True,
)
```

### Q3：ImportError: No module named 'flash_attn'

```
这是正常的。模型会自动 fallback 到标准注意力，同时打印 UserWarning。
功能完全正常，只是速度略慢。
```

### Q4：训练 PQ 指标不上升

```
可能原因：
1. 学习率太大/太小：尝试 lr = 1e-4（默认）到 5e-5 之间
2. batch size 太小：确保总 batch size ≥ 8（4块GPU × 2）
3. 数据预处理问题：检查 FloorPlanCAD-sampled-as-line-jsons 是否正确生成
4. 评估频率太低：减小 eval_steps（默认 2175）更早发现问题
```

### Q5：如何查看训练曲线

```bash
# 启动 TensorBoard
tensorboard --logdir outputs/

# 浏览器打开 http://localhost:6006
# 关注：train/loss, eval/PQ, eval/SQ, eval/RQ
```

### Q6：如何只用 1 块 GPU 训练（用于调试）

```bash
# 修改 scripts/train.sh，或直接运行：
NPROC_PER_NODE=1 bash scripts/train.sh

# 注意：1 GPU 时 batch size 实际只有 2，需要增大 epoch 或减小 eval_steps
```

### Q7：如何可视化预测结果

```bash
# 数据预处理时加 --save_type=svg 可以生成 SVG 文件
python data/floorplancad/preprocess.py \
    --input_dir=datasets/FloorPlanCAD \
    --output_dir=datasets/debug_vis \
    --dynamic_sampling \
    --connect_lines \
    --save_type=svg
```

---

## 附录：推荐学习资源

### PyTorch
- [官方 60 分钟快速入门](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [《动手学深度学习》](https://zh.d2l.ai/)（中文，免费）

### Transformer / 注意力机制
- [The Illustrated Transformer（图解 Transformer）](https://jalammar.github.io/illustrated-transformer/)
- 论文：*Attention Is All You Need*（2017）

### 目标检测/分割
- DETR 论文（本项目 CAD Decoder 的思路来源）：*End-to-End Object Detection with Transformers*（2020）
- Mask2Former（全景分割）论文

### 点云处理
- PointNet 论文（入门点云深度学习）
- PointTransformer V3 论文（本项目骨干网络）
