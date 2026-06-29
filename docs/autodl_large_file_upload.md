# AutoDL large file upload notes

This repository should only store code, configs, and lightweight dataset YAML
files. Do not upload datasets, training runs, checkpoints, or virtual
environments to GitHub.

## Recommended layout on AutoDL

Use `/root/autodl-tmp` for large files:

```bash
mkdir -p /root/autodl-tmp/Graph-MDETR/dataset
mkdir -p /root/autodl-tmp/Graph-MDETR/weights
```

Clone code from GitHub:

```bash
cd /root/autodl-tmp
git clone https://github.com/<user>/<repo>.git Graph-MDETR
cd Graph-MDETR
```

Install dependencies in the AutoDL environment:

```bash
pip install -r requirements.txt
pip install -e .
```

## Upload Det-Fly from Windows

If the AutoDL instance has SSH enabled, upload the prepared dataset directory
with `scp` from PowerShell:

```powershell
scp -r D:\PyCharmPojects\Graph-MDETR\dataset\detfly root@<autodl_ip>:/root/autodl-tmp/Graph-MDETR/dataset/
scp D:\PyCharmPojects\Graph-MDETR\dataset\detfly.yaml root@<autodl_ip>:/root/autodl-tmp/Graph-MDETR/dataset/
```

For a more stable transfer, pack first and then upload one archive:

```powershell
tar -cf D:\BaiduNetdiskDownload\detfly.tar -C D:\PyCharmPojects\Graph-MDETR\dataset detfly detfly.yaml
scp D:\BaiduNetdiskDownload\detfly.tar root@<autodl_ip>:/root/autodl-tmp/Graph-MDETR/dataset/
```

Then extract on AutoDL:

```bash
cd /root/autodl-tmp/Graph-MDETR/dataset
tar -xf detfly.tar
```

## Upload checkpoints

Upload selected checkpoints only:

```powershell
scp D:\PyCharmPojects\Graph-MDETR\run\detfly_smoke_tiny_b1_320_q3\full_k3\weights\best.pt root@<autodl_ip>:/root/autodl-tmp/Graph-MDETR/weights/
```

## Train on AutoDL

Example command:

```bash
cd /root/autodl-tmp/Graph-MDETR
python scripts/run_ablation.py train \
  --only '^full_k3$' \
  --data /root/autodl-tmp/Graph-MDETR/dataset/detfly.yaml \
  --project /root/autodl-tmp/Graph-MDETR/run/detfly_full_k3 \
  --epochs 300 \
  --batch 1 \
  --imgsz 640 \
  --device 0 \
  --workers 4 \
  --save-period 10 \
  --exist-ok \
  --quiet-model
```

