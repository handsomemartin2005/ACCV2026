import argparse
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import RTDETR

ABLATION_DIR = ROOT / 'ultralytics/cfg/models/rt-detr/ablations'


EXPERIMENTS = {
    'baseline': ROOT / 'ultralytics/cfg/models/rt-detr/gcn-rtdetr-mamba.yaml',
    'dtsr_k3': ABLATION_DIR / 'sch_dtsr_k3.yaml',
    'clip_dtsr_k3': ABLATION_DIR / 'sch_clip_dtsr_k3.yaml',
    'dtsr_uhr_k3': ABLATION_DIR / 'sch_dtsr_uhr_k3.yaml',
    'full_k1': ABLATION_DIR / 'sch_full_k1.yaml',
    'full_k2': ABLATION_DIR / 'sch_full_k2.yaml',
    'full_k3': ABLATION_DIR / 'sch_full_k3.yaml',
    'full_k4': ABLATION_DIR / 'sch_full_k4.yaml',
    'full_m4_k3': ABLATION_DIR / 'sch_full_m4_k3.yaml',
    'full_m5_k3': ABLATION_DIR / 'sch_full_m5_k3.yaml',
    'full_m6_k3': ABLATION_DIR / 'sch_full_m6_k3.yaml',
    'no_highfreq_k3': ABLATION_DIR / 'sch_no_highfreq_k3.yaml',
    'no_scale_prior_k3': ABLATION_DIR / 'sch_no_scale_prior_k3.yaml',
    'no_structure_prior_k3': ABLATION_DIR / 'sch_no_structure_prior_k3.yaml',
    'clip_text_only_k3': ABLATION_DIR / 'sch_clip_text_only_k3.yaml',
    'uhr_no_category_k3': ABLATION_DIR / 'sch_uhr_no_category_k3.yaml',
    'uhr_no_bbox_k3': ABLATION_DIR / 'sch_uhr_no_bbox_k3.yaml',
    'uhr_no_structure_k3': ABLATION_DIR / 'sch_uhr_no_structure_k3.yaml',
}


def parse_cuda_index(device):
    device = str(device)
    if device.isdigit():
        return int(device)
    if device.startswith('cuda:') and device.split(':', 1)[1].isdigit():
        return int(device.split(':', 1)[1])
    return None


def apply_gpu_memory_limit(args):
    if not args.gpu_memory_gb or not torch.cuda.is_available():
        return
    device_idx = parse_cuda_index(args.device)
    if device_idx is None:
        return
    total_gb = torch.cuda.get_device_properties(device_idx).total_memory / 1024 ** 3
    fraction = max(0.01, min(float(args.gpu_memory_gb) / total_gb, 1.0))
    torch.cuda.set_per_process_memory_fraction(fraction, device_idx)
    print(f'cuda:{device_idx} per-process memory cap: {args.gpu_memory_gb:.2f} GiB ({fraction:.3f} of {total_gb:.2f} GiB)')


def selected_experiments(pattern):
    rx = re.compile(pattern) if pattern else None
    return [(name, cfg) for name, cfg in EXPERIMENTS.items() if rx is None or rx.search(name)]


def ensure_configs():
    if not ABLATION_DIR.exists() or not any(ABLATION_DIR.glob('*.yaml')):
        subprocess.run([sys.executable, str(ROOT / 'scripts/generate_sch_ablation_configs.py')], check=True)


def run_smoke(name, cfg, args):
    cmd = [
        sys.executable,
        str(ROOT / 'smoke_train_grad.py'),
        '--cfg',
        str(cfg),
        '--imgsz',
        str(args.imgsz),
        '--device',
        args.device,
    ]
    print(' '.join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def run_train(name, cfg, args):
    print(f'train {name}: {cfg}')
    if args.dry_run:
        return
    model = RTDETR(str(cfg), verbose=not args.quiet_model)
    train_kwargs = dict(
        data=str(args.data),
        cache=False,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=name,
        workers=args.workers,
        save_period=args.save_period,
        exist_ok=args.exist_ok,
        verbose=not args.quiet_model,
        deterministic=args.deterministic,
    )
    if args.resume_from:
        train_kwargs['resume'] = str(args.resume_from)
    model.train(**train_kwargs)


def run_val(name, cfg, args):
    weights = args.project / name / 'weights' / args.val_weight
    model_arg = str(weights) if weights.exists() else str(cfg)
    print(f'val {name}: {model_arg}')
    if args.dry_run:
        return
    model = RTDETR(model_arg, verbose=not args.quiet_model)
    model.val(data=str(args.data), imgsz=args.imgsz, batch=args.batch, device=args.device,
              project=str(args.project), name=f'{name}_val', verbose=not args.quiet_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['list', 'smoke', 'train', 'val'])
    parser.add_argument('--only', default='', help='Regex filter for experiment names, e.g. "full_k[1-4]|baseline".')
    parser.add_argument('--data', type=Path, default=ROOT / 'dataset/coco_UAV.yaml')
    parser.add_argument('--project', type=Path, default=ROOT / 'run/ablation')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=224)
    parser.add_argument('--device', default='0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save-period', type=int, default=-1)
    parser.add_argument('--val-weight', default='best.pt')
    parser.add_argument('--resume-from', type=Path, default=None)
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--quiet-model', action='store_true', help='Skip verbose model summary/FLOPs tracing.')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic training. By default this is disabled for speed.')
    parser.add_argument('--gpu-memory-gb', type=float, default=0.0,
                        help='Set a PyTorch CUDA per-process memory cap in GiB. 0 disables the cap.')
    args = parser.parse_args()

    apply_gpu_memory_limit(args)
    ensure_configs()
    experiments = selected_experiments(args.only)
    if args.mode == 'list':
        for name, cfg in experiments:
            print(f'{name}: {cfg}')
        return

    args.project.mkdir(parents=True, exist_ok=True)
    for name, cfg in experiments:
        if args.mode == 'smoke':
            run_smoke(name, cfg, args)
        elif args.mode == 'train':
            run_train(name, cfg, args)
        elif args.mode == 'val':
            run_val(name, cfg, args)


if __name__ == '__main__':
    main()
