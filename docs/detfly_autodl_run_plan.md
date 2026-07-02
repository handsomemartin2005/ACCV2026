# Det-Fly AutoDL Run Plan

Use this note after cloning/pulling the repository on AutoDL. Put the unpacked
dataset at `dataset/detfly/` and keep `dataset/detfly.yaml` pointing to it.

## P0 Training

List available experiment names:

```bash
python scripts/run_ablation.py list --data dataset/detfly.yaml
```

Smoke-check the lightweight Det-Fly configs before long training:

```bash
python scripts/run_ablation.py smoke \
  --only "^(rtdetr_r18_n|sch_rtdetr_r18_n|full_k3_detfly)$" \
  --data dataset/detfly.yaml \
  --imgsz 640 \
  --device 0
```

Train the minimum matched Det-Fly set:

```bash
python scripts/run_ablation.py train \
  --only "^(rtdetr_r18_n|sch_rtdetr_r18_n|rtdetr_r18|full_k3_detfly)$" \
  --data dataset/detfly.yaml \
  --project run/detfly_p0_300e_640 \
  --epochs 300 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --workers 0 \
  --save-period 10 \
  --exist-ok \
  --quiet-model \
  --amp \
  --warmup-epochs 3 \
  --patience 50
```

If memory is tight, rerun with `--batch 8` or add `--gpu-memory-gb 20`.

## P0 Ablation

Run the core module ablations after the main model starts cleanly:

```bash
python scripts/run_ablation.py train \
  --only "^(baseline|dtsr_k3|clip_dtsr_k3|dtsr_uhr_k3|full_k3_detfly)$" \
  --data dataset/detfly.yaml \
  --project run/detfly_ablation_300e_640 \
  --epochs 300 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --workers 0 \
  --save-period 10 \
  --exist-ok \
  --quiet-model \
  --amp \
  --warmup-epochs 3 \
  --patience 50
```

## Visualization

After `best.pt` exists, generate SCH routing, semantic-response, UHR, and
summary visualizations:

```bash
python scripts/visualize_sch.py \
  --cfg ultralytics/cfg/models/rt-detr/sch-rtdetr-r18-n-detfly.yaml \
  --weights run/detfly_p0_300e_640/sch_rtdetr_r18_n/weights/best.pt \
  --source dataset/detfly/images/val \
  --classes dataset/detfly/classes.txt \
  --out run/detfly_p0_300e_640/visualizations/sch_rtdetr_r18_n \
  --imgsz 640 \
  --device cuda:0 \
  --limit 32
```

For the full SCH-MDETR variant, change `--cfg`, `--weights`, and `--out` to the
matching `full_k3_detfly` run directory.
