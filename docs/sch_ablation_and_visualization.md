# SCH-MDETR 消融实验与可视化说明

## 1. TopK 的含义

在当前实现里，`K` 不是 router 自动决定的数量，而是人为设定的路由容量超参。

```text
M 个候选尺度: S1, S2, ..., SM
router 输出: 每个尺度的分数 alpha_1 ... alpha_M
TopK: 根据分数选择 K 个源尺度
```

也就是说：

```text
K 决定最多激活几个源尺度
router 决定具体激活哪几个源尺度
```

为了让 `K=1/2/3/4` 的实验公平可比，代码保持后端 AIFI/FPN/Decoder 的 3 个输出尺度接口不变。不同 `K` 只改变“参与融合的源尺度数量”，不会改变 decoder 输入层数。

因此，验证逻辑应该是两层：

1. **K 消融**：训练 `K=1/2/3/4`，看哪个 K 的 mAP 最好，用来证明路由容量设置合理。
2. **router 可视化**：在最优 K 下统计不同图像的 TopK 选择，看 router 是否随图像内容动态切换尺度。

如果 `K=3` 最好，并且可视化显示不同图像会选择不同的 `S_i` 组合，就可以说明：

```text
K=3 是有效容量，router 学到的是图像自适应尺度选择，而不是固定 S3/S4/S5。
```

## 2. 已实现的可视化

脚本：

```powershell
python .\scripts\visualize_sch.py --cfg ultralytics/cfg/models/rt-detr/sch-rtdetr-mamba.yaml --source dataset/coco_uav/images/val --out run/visualize_sch --imgsz 640 --device 0 --limit 32
```

如果要加载训练权重：

```powershell
python .\scripts\visualize_sch.py --cfg ultralytics/cfg/models/rt-detr/sch-rtdetr-mamba.yaml --weights run/ablation/full_k3/weights/best.pt --source dataset/coco_uav/images/val --out run/visualize_sch_full_k3 --imgsz 640 --device 0 --limit 32
```

输出内容：

- `*_route.png`：原图 + `S1...SM` router 分数柱状图，TopK 尺度高亮。
- `*_semantic.png`：CLIP 类别语义响应条形图。
- `*_uhr.png`：UHR 前后 reference box 中心、query 更新幅度、query 相似度矩阵。
- `route_frequency.png`：验证集上各尺度被选中的频率。
- `route_score_distribution.png`：各尺度 router 分数均值/方差。
- `route_records.csv`：每张图的 route scores、TopK indices、Top semantic classes。

## 3. 已实现的消融配置

生成配置：

```powershell
python .\scripts\generate_sch_ablation_configs.py
```

配置目录：

```text
ultralytics/cfg/models/rt-detr/ablations/
```

主要实验：

| 实验名 | 作用 |
| --- | --- |
| `baseline` | 原 Graph-MDETR / fixed-scale baseline |
| `dtsr_k3` | 只加 DTSR，不用 CLIP，不用 UHR |
| `clip_dtsr_k3` | CLIP + DTSR，不用 UHR |
| `dtsr_uhr_k3` | DTSR + UHR，不用 CLIP |
| `full_k1/2/3/4` | 完整模型的 K 消融 |
| `full_m4/5/6_k3` | 候选尺度数量 M 消融 |
| `no_highfreq_k3` | 去掉高频先验 |
| `no_scale_prior_k3` | 去掉尺度统计/尺度嵌入先验 |
| `no_structure_prior_k3` | 去掉 router 结构先验 |
| `clip_text_only_k3` | 只用 CLIP text prototype，不用 CLIP image prior |
| `uhr_no_category_k3` | 去掉 UHR 类别分支 |
| `uhr_no_bbox_k3` | 去掉 UHR bbox 几何分支 |
| `uhr_no_structure_k3` | 去掉 UHR 结构分支 |

## 4. 运行消融

列出实验：

```powershell
python .\scripts\run_ablation.py list
```

先 smoke 检查：

```powershell
python .\scripts\run_ablation.py smoke --only "^(baseline|dtsr_k3|clip_dtsr_k3|full_k[1-4])$" --imgsz 224 --device cpu
```

训练主消融：

```powershell
python .\scripts\run_ablation.py train --only "^(baseline|dtsr_k3|clip_dtsr_k3|dtsr_uhr_k3|full_k3)$" --epochs 300 --batch 16 --imgsz 640 --device 0
```

训练 K 消融：

```powershell
python .\scripts\run_ablation.py train --only "^full_k[1-4]$" --epochs 300 --batch 16 --imgsz 640 --device 0
```

训练 M 消融：

```powershell
python .\scripts\run_ablation.py train --only "^full_m[4-6]_k3$" --epochs 300 --batch 16 --imgsz 640 --device 0
```

## 5. 汇总结果

训练完成后：

```powershell
python .\scripts\plot_ablation_results.py --project run/ablation --out run/ablation/summary
```

输出：

- `ablation_summary.csv`
- `metric_curves.png`
- `loss_curves.png`
- `map50_95_bar.png`

