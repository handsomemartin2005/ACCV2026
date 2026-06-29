# SCH-MDETR 方法设计文档

## 1. 一句话概括

**SCH-MDETR 是在 Mamba-DETR / DETR 检测框架上，引入 CLIP 语义先验、动态尺度路由和统一超图推理，用于解决 UAV 接触网小目标检测中的尺度变化、语义混淆和结构关系建模不足问题。**

整体主线为：

```text
Input UAV Image
    |
CLIP-guided Semantic Prototype Learning
    |
Dynamic Top-K Scale Router
    |
Efficient Hybrid Encoder / DAMamba
    |
Unified Hypergraph Reasoning
    |
Transformer Decoder & Detection Head
    |
Detection Results
```

可以将核心逻辑概括为：

```text
语义引导 -> 动态尺度选择 -> 高阶结构推理 -> 精确检测
```

## 2. 核心创新链

SCH-MDETR 包含三个核心创新点：

| 创新点 | 加入位置 | 创新内容 | 解决问题 |
| --- | --- | --- | --- |
| **CLIP-guided Semantic Prototype Learning** | 网络前端，指导 DTSR 和 UHR | 使用 CLIP Text Encoder 生成类别语义原型，并使用 CLIP-ViT 提取图像语义先验 | 缓解小目标纹理弱、类别相似、语义混淆 |
| **Dynamic Top-K Scale Router** | 多尺度候选特征之后，Hybrid Encoder 之前 | 根据图像内容、语义响应、频率信息、尺度统计和结构先验动态选择 Top-K 尺度 | 缓解固定尺度融合导致的尺度错配和特征冗余 |
| **Unified Hypergraph Reasoning** | Initial queries / reference boxes 之后，Transformer Decoder 之前 | 将 query、CLIP prototype、bbox geometry 和 structural priors 统一构造成超图进行高阶推理 | 缓解 query 孤立、语义关系和几何结构分离建模的问题 |

## 3. CLIP-Guided Semantic Prototype Learning

### 3.1 加入位置

CLIP-guided Semantic Prototype Learning 位于检测网络前端，同时为后续两个模块提供语义先验：

- 为 **Dynamic Top-K Scale Router** 提供图像级和类别级语义提示，用于尺度选择。
- 为 **Unified Hypergraph Reasoning** 提供类别原型节点，用于 query、类别、几何和结构关系的统一推理。

其流程可以表示为：

```text
Category Names
    |
CLIP Text Encoder
    |
CLIP Category Prototypes
```

以及：

```text
Input Image
    |
CLIP-ViT Image Encoder
    |
CLIP Visual Prior
```

### 3.2 创新内容

该模块不是简单地将 backbone 替换为 CLIP，而是将 CLIP 拆分为两类先验使用：

```text
1. 文本类别原型：提供类别级语义先验
2. 图像语义特征：提供视觉语义响应
```

对于第 `c` 个类别，其类别名称为 `t_c`，通过 CLIP Text Encoder 得到类别文本原型：

```math
p_c = E_{\text{text}}^{\text{CLIP}}(t_c)
```

其中，`p_c` 表示第 `c` 类的 CLIP 文本原型。

输入图像 `I` 经过 CLIP image encoder 后得到视觉 token：

```math
v_i = E_{\text{image}}^{\text{CLIP}}(I)_i
```

视觉 token 与类别原型之间的语义相似度可以写为：

```math
s_{i,c} =
\frac{v_i^\top p_c}{\|v_i\| \|p_c\|}
```

其中，`s_{i,c}` 表示第 `i` 个视觉 token 对第 `c` 类语义原型的响应。该语义响应可以用于指导后续的尺度选择和超图推理。

### 3.3 解决的问题

UAV 接触网支撑部件检测中，许多目标具有以下特点：

- 目标尺寸小。
- 纹理细节弱。
- 类别之间外观相似。
- 目标之间存在遮挡和结构依赖。

例如绝缘子、夹具、支撑结构、销钉等部件，在局部纹理上容易混淆。CLIP 类别原型的作用是让模型不仅依赖局部视觉纹理，还能利用类别语义先验辅助识别。

## 4. Dynamic Top-K Scale Router

### 4.1 加入位置

Dynamic Top-K Scale Router，简称 **DTSR**，位于多尺度候选特征生成之后、Efficient Hybrid Encoder 之前：

```text
Candidate Multi-scale Features
    |
Dynamic Top-K Scale Router
    |
Selected Top-K Scale Features
    |
Efficient Hybrid Encoder / DAMamba
```

传统检测框架通常固定使用 `P3, P4, P5` 或 `S3, S4, S5` 等尺度。DTSR 将这一固定规则改为图像自适应选择：

```text
固定规则：S1-S5 -> S3,S4,S5
动态规则：S1-SM -> TopK(S1...SM)
```

### 4.2 候选尺度特征

假设候选尺度特征为：

```math
\mathcal{F} = \{F_1, F_2, \cdots, F_M\}
```

其中，`F_m` 表示第 `m` 个候选尺度的特征，`M` 是候选尺度数量。`M` 不必固定为 5，可以根据结构设计为 5、6、7 或更多。

DTSR 首先对每个尺度提取描述向量：

```math
z_m = \text{Pool}(F_m)
```

`z_m` 表示第 `m` 个尺度的全局描述。

### 4.3 路由先验

DTSR 的尺度评分不应只依赖单个尺度的池化特征，还可以融合多种先验：

```text
Global Semantic
High Frequency
Category Co-occurrence
Scale Statistics
Structural Topology
```

这些先验分别对应：

- **Global Semantic**：判断图像整体语义复杂度。
- **High Frequency**：感知边缘、纹理和细长小目标。
- **Category Co-occurrence**：利用类别共现关系辅助选择有效尺度。
- **Scale Statistics**：估计当前图像中的目标尺度倾向。
- **Structural Topology**：利用接触网部件之间的结构关系。

如果引入 CLIP 和结构图先验，可以将尺度评分写为：

```math
\alpha_m = \text{Router}(z_m, P, R)
```

其中：

- `P` 表示 CLIP 类别原型或 CLIP 语义响应。
- `R` 表示结构先验。
- `\alpha_m` 表示第 `m` 个尺度的重要性分数。

### 4.4 Top-K 选择

根据所有尺度分数得到 Top-K 尺度集合：

```math
\mathcal{S} = \text{TopK}(\alpha, K)
```

最终得到固定数量的路由特征：

```math
\mathcal{Y} = \{Y_1, Y_2, \cdots, Y_K\}
```

如果设置 `K = 3`，则输出为：

```text
Y1, Y2, Y3
```

这意味着 decoder 或 hybrid encoder 的输入数量仍然固定为 3，但这 3 个尺度不再固定为 `S3,S4,S5`，而是随图像内容动态变化。

### 4.5 它不是强化学习

DTSR 不是 reinforcement learning，也不需要 reward、policy 或额外的策略优化。它通过检测损失反向传播进行端到端训练。

检测损失可以写为：

```math
\mathcal{L}_{det}
=
\mathcal{L}_{cls}
+
\lambda_1 \mathcal{L}_{box}
+
\lambda_2 \mathcal{L}_{giou}
```

router 参数通过梯度下降更新：

```math
\theta_{router}
\leftarrow
\theta_{router}
-
\eta
\frac{\partial \mathcal{L}_{det}}
{\partial \theta_{router}}
```

因此，图中的反向更新箭头建议标注为：

```text
Detection-loss-driven gradient update
```

不要标注为 reward、policy 或 RL。

### 4.6 训练方式

训练初期建议使用 soft gate，而不是直接 hard Top-K：

```math
g_m = \sigma(\alpha_m)
```

```math
\tilde{F}_m = g_m F_m
```

这样每个尺度都有连续梯度，检测损失可以稳定反向传播到 router。训练稳定后，再逐步引入 Top-K mask：

```math
M_m =
\begin{cases}
1, & m \in \text{TopK}(\alpha, K) \\
0, & \text{otherwise}
\end{cases}
```

```math
\tilde{F}_m = M_m g_m F_m
```

推理阶段可以只保留 Top-K 特征进入后续模块，从而减少冗余尺度输入。

### 4.7 解决的问题

UAV 接触网小目标检测中，目标尺度变化大，固定尺度融合存在两个问题：

```text
1. 有些尺度对当前图像无效，会带来冗余计算和噪声。
2. 小目标可能只在某些高分辨率尺度上明显，固定使用 S3-S5 容易出现尺度错配。
```

DTSR 的作用是将固定多尺度融合改为图像自适应尺度选择。

## 5. Unified Hypergraph Reasoning

### 5.1 加入位置

Unified Hypergraph Reasoning，简称 **UHR**，位于 initial queries / reference boxes 之后、Transformer Decoder 之前：

```text
Initial Queries / Reference Boxes
    |
Unified Hypergraph Reasoning
    |
Enhanced Queries / Reference Boxes
    |
Transformer Decoder & Detection Head
```

它不是普通后处理模块，而是 decoder 前的 query 和 bbox 增强模块。

### 5.2 创新内容

原始 DETR 流程通常是：

```text
Initial Queries / Reference Boxes
    |
Transformer Decoder
```

SCH-MDETR 将其改为：

```text
Initial Queries / Reference Boxes
    |
CLIP Prototypes + Structural Priors
    |
Unified Hypergraph Reasoning
    |
Enhanced Queries / Reference Boxes
    |
Transformer Decoder
```

UHR 的输入包含四类信息：

```text
1. Query features
2. CLIP category prototypes
3. BBox geometry
4. Structural priors
```

可以写为：

```math
X = [Q, P, B, R]
```

其中：

- `Q` 表示 object queries。
- `P` 表示 CLIP 类别原型。
- `B` 表示 reference boxes。
- `R` 表示结构先验。

### 5.3 统一超图构建

构造统一超图：

```math
\mathcal{G}_h = (\mathcal{V}, \mathcal{E})
```

节点集合为：

```math
\mathcal{V}
=
\mathcal{V}_q
\cup
\mathcal{V}_p
\cup
\mathcal{V}_b
\cup
\mathcal{V}_r
```

其中：

- `\mathcal{V}_q`：query 节点。
- `\mathcal{V}_p`：category prototype 节点。
- `\mathcal{V}_b`：bbox geometry 节点。
- `\mathcal{V}_r`：structural prior 节点。

普通图通常表达两个节点之间的关系：

```text
query_i -- query_j
```

而超图的一条超边可以连接多个节点：

```text
query_i -- category_c -- bbox_i -- structure_k
```

因此，UHR 可以同时建模 query、类别语义、bbox 几何和结构拓扑之间的高阶关系。

### 5.4 与 SGM / PGM 的区别

如果原方法使用：

```text
SGM: Semantic Graph Module
PGM: Position / Physical Graph Module
```

那么语义关系和几何关系是分开建模的。UHR 的核心变化是：

```text
不再将 SGM 和 PGM 分离建模，而是将 Query、Category、BBox 和 Structure 统一放入一个超图中推理。
```

这使得模型可以在 decoder 之前联合捕获语义、几何和拓扑关系。

### 5.5 解决的问题

接触网目标不是孤立分布的，而是存在明显结构关系：

```text
绝缘子通常靠近支撑结构。
夹具通常依附于线缆。
导线具有连续方向。
不同部件之间存在稳定空间拓扑。
```

普通 DETR query 之间缺少显式结构约束。UHR 的作用是让 query 在进入 decoder 前，先融合类别语义、bbox 几何和结构拓扑关系。

最终输出可以写为：

```math
Q^+, B^+ = \text{UHR}(Q, B, P, R)
```

然后送入 decoder：

```math
\hat{Y}
=
\text{Head}(\text{Decoder}(Q^+, B^+, Z))
```

## 6. Efficient Hybrid Encoder / DAMamba

Efficient Hybrid Encoder / DAMamba 位于 DTSR 之后、UHR 之前：

```text
Selected Top-K Scale Features
    |
Efficient Hybrid Encoder / DAMamba
    |
Encoded Multi-scale Feature Z
```

其作用是对路由后的尺度特征进行编码和融合：

```math
Z = \text{HybridEncoder}(Y_1, Y_2, Y_3)
```

它可以包括：

```text
AIFI
CBS
Fusion
DAMamba Block
```

如果 DAMamba 是在原方法基础上进一步改造得到的，可以作为辅助创新点展开；如果主要沿用已有 Mamba-DETR / Graph-MDETR 结构，则建议将其表述为主干增强模块，而不是最核心创新。

更稳妥的论文表述为：

```text
We employ an efficient hybrid encoder with DAMamba blocks to enhance routed multi-scale features.
```

## 7. 论文贡献写法

### Contribution 1

提出 **CLIP-guided Semantic Prototype Learning**，利用类别文本原型和图像语义响应为 UAV 小目标检测提供语义先验，增强模型对弱纹理、易混淆接触网部件的类别辨识能力。

### Contribution 2

提出 **Dynamic Top-K Scale Router**，根据当前图像的语义、频率、尺度和结构特征动态选择最有效的 `K` 个尺度特征，实现图像自适应的多尺度特征路由，缓解固定尺度融合导致的尺度错配和特征冗余问题。

### Contribution 3

提出 **Unified Hypergraph Reasoning**，将 object queries、CLIP 类别原型、reference boxes 和结构先验统一建模为超图，通过高阶消息传递联合捕获语义、几何和拓扑关系，替代原先分离式的 SGM / PGM 建模方式。

## 8. 精炼版本

这篇方法可以压缩成三句话：

```text
1. CLIP 负责提供类别语义先验，让模型知道“检测的目标在语义上是什么”。

2. DTSR 负责动态选择尺度，让模型知道“当前图像应该重点看哪些尺度”。

3. UHR 负责统一关系推理，让模型知道“目标之间在语义、几何和结构上有什么关系”。
```

最终形成的创新链为：

```text
语义引导 -> 动态尺度选择 -> 高阶结构推理 -> 精确检测
```

## 9. 实现优先级建议

建议按以下顺序落地：

1. **先实现 DTSR**：从 `M` 个候选尺度中动态选 `Top-K`，训练阶段使用 soft gate，推理阶段使用 hard Top-K。
2. **再引入 CLIP prototype**：先使用 CLIP text prototype 指导类别语义，再考虑 CLIP image prior。
3. **最后实现 UHR**：先构造 query-category-bbox 的简化超图，再加入完整结构先验。

这样可以降低工程风险，并且方便做消融实验：

```text
Baseline
Baseline + DTSR
Baseline + DTSR + CLIP
Baseline + DTSR + CLIP + UHR
```
