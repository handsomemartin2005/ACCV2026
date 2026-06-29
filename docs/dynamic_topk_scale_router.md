# Dynamic Top-K Scale Router 设计文档

## 1. 背景与问题

在当前 DAMamba-Adapter 结构中，论文图示通常将 backbone 的层级输出表示为 `S1-S5`，并在后续的 cross-scale fusion 或 decoder 中选取部分尺度特征，例如常见的 `S3, S4, S5`。这种设计本质上是一种人工设定：

- 候选特征层数固定为 5。
- 进入 decoder 的尺度固定为 3 个。
- 默认所有图像都适合同一组尺度组合。

但在 UAV 接触网支撑部件检测任务中，不同图像的目标尺度、遮挡程度、背景复杂度和细节分布差异很大。固定选取 `S3-S5` 可能不总是最优：

- 小目标、细长零件较多时，高分辨率浅层特征更重要。
- 目标较大、语义清晰时，深层特征更有效。
- 背景纹理复杂时，浅层细节可能既包含有用边缘，也包含噪声。
- 遮挡严重或类别共现明显时，图结构先验可能影响需要保留的尺度层。

因此，可以将固定的 `S1-S5` 设计扩展为：

```text
从 M 个候选尺度特征中，动态选择 Top-K 个最有用尺度进入后续 encoder / decoder。
```

即：

```text
固定规则：S1-S5 -> S3,S4,S5

改进规则：S1-SM -> TopK(S1...SM)
```

## 2. 核心思想

提出一个动态尺度选择模块：

```text
Dynamic Top-K Scale Router, DTSR
```

该模块位于 DAMamba-Adapter 输出之后、cross-scale fusion 或 decoder 之前。它首先构建一个包含 `M` 个候选尺度的特征集合，然后根据图像先验和图结构先验为每个尺度打分，最后选出 Top-K 个尺度特征用于检测。

整体流程如下：

```text
Input Image
    |
DAMamba-Adapter
    |
Candidate Scale Bank: F1, F2, ..., FM
    |
Prior Encoder
    |
Scale Router -> score1, score2, ..., scoreM
    |
Top-K Selector
    |
Selected Features -> CCF / Encoder / Decoder
```

## 3. 模块组成

### 3.1 Candidate Scale Bank

`Candidate Scale Bank` 用于生成 `M` 个候选尺度特征：

```text
F = {F1, F2, ..., FM}
```

这里的 `M` 不再必须等于 5，而是一个可配置超参数。候选尺度可以来自三类来源：

1. DAMamba-Adapter 原始输出层。
2. 对已有层进行上采样或下采样生成补充尺度。
3. 使用不同 dilation、pooling 或轻量卷积构造中间尺度。

例如：

```text
M = 6
strides = [4, 8, 12, 16, 24, 32]
```

或：

```text
M = 7
strides = [2, 4, 8, 16, 32, 64, 128]
```

候选层不一定全部进入 decoder，而是先作为可选择的尺度池。

### 3.2 Prior Encoder

`Prior Encoder` 用于提取决定尺度选择的图像先验。它不直接输出检测结果，而是为 router 提供判断依据。

可以使用以下信号：

#### 3.2.1 全局语义特征

全局语义特征用于判断图像整体内容和上下文复杂度。若图像中目标较大、语义清晰，深层特征通常更有效；若目标分布密集或类别混杂，则可能需要更多中浅层信息辅助。

一种简单实现：

```text
global_semantic = GAP(Conv(F_deep))
```

#### 3.2.2 边缘与高频信息

接触网支撑部件中存在许多小目标和细长结构，例如螺母、线夹、销钉、定位器等。这类目标的边界、纹理和形状细节往往集中在高频区域。

如果图像高频响应强，说明浅层或中层高分辨率特征可能更重要。

可选实现：

```text
edge_prior = GAP(Sobel(F_shallow))
```

或：

```text
high_freq = GAP(F_shallow - AvgPool(F_shallow))
```

#### 3.2.3 浅层纹理复杂度

浅层特征既包含有用细节，也容易包含背景噪声。纹理复杂度可以帮助 router 判断浅层特征是有利于检测，还是会引入干扰。

例如：

```text
texture_prior = GAP(Conv3x3(F_shallow))
```

#### 3.2.4 目标尺度先验

尺度选择最直接依赖目标大小。小目标更依赖高分辨率特征，大目标更依赖深层语义特征。

训练阶段不能使用真实框作为输入，因此目标尺度先验应由网络从图像特征中预测，例如 objectness、saliency 或 scale response：

```text
scale_prior = MLP(GAP(F1), GAP(F2), ..., GAP(FM))
```

#### 3.2.5 图结构先验

Graph-MDETR 已经包含 SGM / PGM，用于建模类别共现关系和空间位置关系。该先验也可以用于尺度选择：

- 某些类别通常较小，需要浅层高分辨率特征。
- 某些类别遮挡严重，需要中深层上下文。
- 某些空间关系明显的部件需要更强的跨尺度语义融合。

因此可以将图结构特征输入 router：

```text
graph_prior = GraphEncoder(Adj, CategoryEmbedding, PositionPrior)
```

最终先验向量可以表示为：

```text
P = concat(global_semantic, high_freq, texture_prior, scale_prior, graph_prior)
```

## 4. Scale Router

`Scale Router` 根据先验向量 `P` 预测每个候选尺度的重要性分数：

```text
score = MLP(P)
score in R^(B x M)
```

其中：

```text
score_i 表示第 i 个候选尺度 Fi 对当前图像的重要性。
```

然后将分数转为连续 gate：

```text
gate = sigmoid(score)
```

训练阶段推荐使用 soft gate：

```text
Fi' = gate_i * Fi
```

这样所有尺度都能参与训练，检测 loss 可以通过连续 gate 反向传播到 router。

## 5. Top-K Selector

推理阶段可以根据分数选择 Top-K 个尺度：

```text
idx = topk(score, K)
selected_features = {Fi | i in idx}
```

如果 `K` 固定，例如从 `M=6` 个候选尺度中选择 `K=3` 个，则工程实现相对稳定：

```text
S1-S6 -> Top3
```

如果希望 `K` 也动态变化，可以增加一个 `Scale Count Predictor`：

```text
K = CountPredictor(P)
```

但动态 `K` 会导致 batch 内输入层数不一致，decoder、loss 和导出逻辑都更复杂。因此建议第一阶段使用固定 `K`，只让被选中的尺度组合动态变化。

## 6. 检测 Loss 如何训练 Router

该模块不需要额外标注“某张图应该选哪些尺度”。训练信号来自最终检测损失：

```text
loss = classification loss + bbox loss + giou loss
```

训练路径为：

```text
检测结果
   |
detection loss
   |
decoder / encoder
   |
weighted multi-scale features
   |
gate scores
   |
Scale Router
```

也就是说，如果某个尺度对检测有帮助，增大它的 gate 会降低 loss，梯度会推动 router 在类似图像上给该尺度更高分；如果某个尺度引入噪声或帮助较小，梯度会推动 router 降低该尺度权重。

以小目标图像为例：

```text
初始 router 输出：
S1: 0.15
S2: 0.20
S3: 0.35
S4: 0.80
S5: 0.90
```

模型更依赖深层 `S4/S5`，但小目标在深层特征中已经被严重下采样，检测出现漏检，loss 较大。反向传播后，router 会逐渐提高浅层和中层特征权重：

```text
更新后 router 输出：
S1: 0.30
S2: 0.72
S3: 0.86
S4: 0.68
S5: 0.40
```

此时 Top-K 可能选择：

```text
S2, S3, S4
```

而不是固定的：

```text
S3, S4, S5
```

对于目标较大、背景干净的图像，router 可能输出：

```text
S1: 0.10
S2: 0.25
S3: 0.55
S4: 0.88
S5: 0.91
```

此时 Top-K 仍可能选择：

```text
S3, S4, S5
```

因此，该模块可以根据图像内容自适应选择尺度组合。

## 7. 推荐训练策略

### 7.1 阶段一：Soft Gate 训练

训练初期保留所有候选尺度，只对每个尺度加权：

```text
Fi' = gate_i * Fi
```

优点：

- 连续可导，训练稳定。
- 每个尺度都有梯度。
- router 可以逐渐学习尺度偏好。

### 7.2 阶段二：Top-K 稀疏化

训练中后期引入 Top-K mask：

```text
mask_i = 1 if i in TopK(score) else 0
Fi' = mask_i * gate_i * Fi
```

可以使用 straight-through estimator 或 soft-to-hard annealing，使训练逐渐接近推理时的 hard selection。

### 7.3 阶段三：推理 Hard Top-K

推理阶段只保留 Top-K 个尺度进入后续模块：

```text
selected = TopK(score, K)
```

这样可以减少冗余尺度输入，并实现真正的动态尺度选择。

## 8. 与 Graph-MDETR 的结合方式

该模块可以作为 DAMamba-Adapter 和后续 CCF / decoder 之间的插入模块：

```text
DAMamba-Adapter
    |
Candidate Scale Bank
    |
Dynamic Top-K Scale Router
    |
Selected Multi-scale Features
    |
CCF / AIFI / GUIDE_RTDETRDecoder
```

若结合 Graph-MDETR 的图结构先验，可进一步扩展为：

```text
Graph-guided Dynamic Top-K Scale Router, G-DTSR
```

其评分函数可以写为：

```text
score = Router(Fi, P_img, P_graph)
```

其中：

- `Fi` 表示第 i 个尺度特征。
- `P_img` 表示图像先验。
- `P_graph` 表示 SGM / PGM 提供的图结构先验。

论文表述可写为：

```text
Unlike previous manually designed hierarchical feature selection strategies,
the proposed G-DTSR constructs an M-scale candidate feature bank and dynamically
selects the most informative Top-K scales according to image-aware and graph-guided priors.
```

## 9. 工程落地建议

第一版建议不要直接做动态 `K`，而是：

```text
M 固定，K 固定，Top-K 组合动态。
```

例如：

```text
M = 6
K = 3
```

这样可以保持 decoder 输入数量固定，避免 batch 对齐、通道配置和 ONNX 导出问题。

推荐实现步骤：

1. 修改 DAMamba-Adapter，使其输出候选尺度列表 `features = [F1, ..., FM]`。
2. 新增 `CandidateScaleBank`，用于补充生成更多候选尺度。
3. 新增 `DynamicTopKScaleRouter`，输出 `scores` 和 `gates`。
4. 训练阶段使用 soft gate 加权所有候选尺度。
5. 推理阶段使用 Top-K 选择固定数量的尺度输入 decoder。
6. 保持 `GUIDE_RTDETRDecoder` 接收的尺度数量为固定 `K`。

## 10. 消融实验设计

建议至少做以下实验：

| 实验项 | 目的 |
|---|---|
| 固定 `S3-S5` | baseline |
| `M=5, K=3` dynamic Top-K | 验证动态选择是否优于固定选择 |
| `M=6, K=3` dynamic Top-K | 验证增加候选尺度是否有效 |
| soft gate only | 验证连续加权是否有效 |
| hard Top-K | 验证稀疏选择是否有效 |
| without high-frequency prior | 验证高频先验贡献 |
| without graph prior | 验证 SGM / PGM 先验贡献 |
| different K | 分析输入尺度数量对精度和速度的影响 |

重点指标：

- mAP50
- mAP50:95
- 小目标 AP
- FPS
- FLOPs
- 参数量
- 不同尺度目标的召回率

## 11. 可能风险

### 11.1 Hard Top-K 不可导

如果训练初期直接 hard Top-K，未选中的尺度几乎没有梯度，router 容易训练不稳定。建议先 soft gate，再逐步稀疏化。

### 11.2 Router 坍缩

router 可能长期偏向某几个尺度，导致其他尺度几乎不被使用。可加入负载均衡或熵正则：

```text
L_balance = encourage average gate distribution not collapse
```

### 11.3 候选尺度过多导致计算增加

虽然最终只选 Top-K，但如果所有候选尺度都完整生成，训练阶段计算量会增加。可先从 `M=5` 或 `M=6` 开始。

### 11.4 动态 K 工程复杂

动态 `K` 会带来 batch 内输入数量不一致的问题。建议第一阶段固定 `K`，后续再探索动态 `K`。

## 12. 总结

该想法的核心创新不是简单地增加特征层数，而是将原本人工固定的尺度选择：

```text
S1-S5 -> S3,S4,S5
```

改为图像感知、图结构引导的动态尺度路由：

```text
S1-SM -> TopK(S1...SM)
```

它能够根据不同图像的目标尺度、边缘细节、纹理复杂度和图结构关系，自适应选择最适合当前图像的多尺度特征组合。对于 UAV 接触网支撑部件检测任务，该模块有望提升小目标、遮挡目标和复杂背景下的检测性能，同时减少固定尺度选择带来的冗余与不适配问题。
