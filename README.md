# SCH-MDETR / ACCV2026

Scale-adaptive CLIP-guided Hypergraph Mamba-DETR for visible-light UAV detection.

This repository contains the experimental code used for SCH-MDETR, including:

- Dynamic Top-K Scale Routing (DTSR) over Mamba adapter feature levels.
- CLIP-guided semantic priors for scale scoring.
- Unified Hypergraph Reasoning (UHR) in the RT-DETR decoder.
- Det-Fly dataset preparation scripts.
- Ablation, smoke-test, and visualization utilities.

Large files are intentionally not tracked. Datasets, training runs, checkpoints,
virtual environments, and archives are excluded by `.gitignore`.

## Repository layout

```text
docs/                       Method notes, ablation design, AutoDL upload guide
scripts/                    Dataset, training, ablation, and visualization tools
dataset/*.yaml              Lightweight dataset config files only
ultralytics/cfg/models/     RT-DETR and SCH-MDETR model configs
ultralytics/nn/modules/     SCH routing and decoder modules
ultralytics/nn/backbone/    DAMamba/Mamba backbone code
```

## Environment

Windows PowerShell:

```powershell
cd D:\PyCharmPojects\Graph-MDETR
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Install the local Mamba package:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_local_mamba.ps1
```

On AutoDL/Linux, clone the repository and install in the target Python
environment:

```bash
cd /root/autodl-tmp
git clone https://github.com/handsomemartin2005/ACCV2026.git Graph-MDETR
cd Graph-MDETR
pip install -r requirements.txt
pip install -e .
```

## Det-Fly dataset

Download Det-Fly images and annotations separately. The prepared dataset is not
committed to GitHub.

Expected raw download layout on Windows:

```text
D:\BaiduNetdiskDownload\Annotations
D:\BaiduNetdiskDownload\JPEGImages
```

Prepare the project-local Ultralytics dataset:

```powershell
python scripts\prepare_detfly_dataset.py `
  --annotations D:\BaiduNetdiskDownload\Annotations `
  --images D:\BaiduNetdiskDownload\JPEGImages `
  --out D:\PyCharmPojects\Graph-MDETR\dataset\detfly `
  --yaml D:\PyCharmPojects\Graph-MDETR\dataset\detfly.yaml `
  --mode hardlink
```

The generated dataset uses:

```text
dataset/detfly/images/{train,val,test}
dataset/detfly/labels/{train,val,test}
dataset/detfly.yaml
```

Class names:

```text
0: UAV
```

## Smoke test

Run a tiny batch-size-1 smoke test:

```powershell
python scripts\run_ablation.py train `
  --only ^full_k3$ `
  --data D:\PyCharmPojects\Graph-MDETR\dataset\detfly_smoke_tiny.yaml `
  --project D:\PyCharmPojects\Graph-MDETR\run\detfly_smoke_tiny_b1_320_q3 `
  --epochs 1 `
  --batch 1 `
  --imgsz 320 `
  --device 0 `
  --workers 0 `
  --save-period 1 `
  --exist-ok `
  --quiet-model
```

`--quiet-model` avoids slow model graph tracing and FLOPs probing for the Mamba
modules during quick tests.

## Training

Run the main full model on Det-Fly:

```powershell
python scripts\run_ablation.py train `
  --only ^full_k3$ `
  --data D:\PyCharmPojects\Graph-MDETR\dataset\detfly.yaml `
  --project D:\PyCharmPojects\Graph-MDETR\run\detfly_full_k3 `
  --epochs 300 `
  --batch 1 `
  --imgsz 640 `
  --device 0 `
  --workers 4 `
  --save-period 10 `
  --exist-ok `
  --quiet-model
```

For quick timing checks, reduce `--epochs` to 10.

## Ablation experiments

Generate ablation configs:

```powershell
python scripts\generate_sch_ablation_configs.py
```

Run selected experiments:

```powershell
python scripts\run_ablation.py train `
  --only "full_k1|full_k2|full_k3|full_k4" `
  --data D:\PyCharmPojects\Graph-MDETR\dataset\detfly.yaml `
  --project D:\PyCharmPojects\Graph-MDETR\run\detfly_ablation_topk `
  --epochs 300 `
  --batch 1 `
  --imgsz 640 `
  --device 0 `
  --workers 4 `
  --quiet-model
```

The ablation registry currently includes baseline, DTSR-only, CLIP+DTSR,
DTSR+UHR, full Top-K variants, dynamic M variants, prior removals, and UHR
component removals.

## Visualization

Create SCH route, semantic, hypergraph, and UHR visualizations from trained
weights:

```powershell
python scripts\visualize_sch.py `
  --cfg ultralytics/cfg/models/rt-detr/ablations/sch_full_k3.yaml `
  --weights D:\PyCharmPojects\Graph-MDETR\run\detfly_full_k3\full_k3\weights\best.pt `
  --source D:\PyCharmPojects\Graph-MDETR\dataset\detfly\images\test `
  --out D:\PyCharmPojects\Graph-MDETR\run\detfly_visualize_full_k3 `
  --imgsz 640 `
  --device cuda:0 `
  --limit 16 `
  --classes dataset\detfly_classes.txt
```

Useful outputs include:

- `*_route.png`: route scores and selected feature levels.
- `*_semantic.png`: CLIP semantic response.
- `*_hypergraph_feature.png`: hypergraph feature activation map.
- `*_uhr_explainable.png`: input, UHR heatmap, query update arrows, and target crop.

## AutoDL large files

Use GitHub for code only. Upload Det-Fly and checkpoints to AutoDL separately,
preferably under `/root/autodl-tmp/Graph-MDETR`.

See [docs/autodl_large_file_upload.md](docs/autodl_large_file_upload.md).

Example upload from Windows:

```powershell
scp -r D:\PyCharmPojects\Graph-MDETR\dataset\detfly root@<autodl_ip>:/root/autodl-tmp/Graph-MDETR/dataset/
scp D:\PyCharmPojects\Graph-MDETR\dataset\detfly.yaml root@<autodl_ip>:/root/autodl-tmp/Graph-MDETR/dataset/
```

## Notes

- The repository tracks code and lightweight configs only.
- The default branch is `main`.
- The main remote is `https://github.com/handsomemartin2005/ACCV2026.git`.
- Formal training results should be generated from full Det-Fly training, not
  from smoke-test checkpoints.

