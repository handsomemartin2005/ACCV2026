import argparse
import csv
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ablation import EXPERIMENTS, ensure_configs


DEFAULT_ORDER = [
    'yolov8_detr_n',
    'yolov8_detr_p2_n',
    'yolov8_detr_fasternet_n',
    'rtdetr_r18_n',
    'rtdetr_r18',
    'baseline',
    'full_k3',
    'dtsr_k3',
    'clip_dtsr_k3',
    'dtsr_uhr_k3',
    'full_k1',
    'full_k2',
    'full_k4',
    'full_m4_k3',
    'full_m5_k3',
    'full_m6_k3',
    'no_highfreq_k3',
    'no_scale_prior_k3',
    'no_structure_prior_k3',
    'clip_text_only_k3',
    'uhr_no_category_k3',
    'uhr_no_bbox_k3',
    'uhr_no_structure_k3',
]


def now():
    return datetime.now().isoformat(timespec='seconds')


def selected_names(pattern):
    rx = re.compile(pattern) if pattern else None
    return [name for name in DEFAULT_ORDER if name in EXPERIMENTS and (rx is None or rx.search(name))]


def results_epochs(run_dir):
    csv_path = run_dir / 'results.csv'
    if not csv_path.exists():
        return 0
    try:
        return len(pd.read_csv(csv_path))
    except Exception:
        return 0


def is_complete(run_dir, epochs):
    return (run_dir / 'weights' / 'best.pt').exists() and results_epochs(run_dir) >= epochs


def append_status(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = [
        'time',
        'stage',
        'name',
        'status',
        'batch',
        'returncode',
        'seconds',
        'epochs_done',
        'log',
    ]
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_logged(cmd, log_path, cwd=ROOT):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open('ab') as log:
        header = f"\n\n===== {now()} =====\nCMD: {' '.join(map(str, cmd))}\n\n"
        log.write(header.encode('utf-8', errors='replace'))
        log.flush()
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        returncode = proc.wait()
        footer = f"\n===== finished {now()} returncode={returncode} seconds={time.time() - started:.1f} =====\n"
        log.write(footer.encode('utf-8', errors='replace'))
    return returncode, time.time() - started


def train_one(name, args, status_path):
    run_dir = args.project / name
    if is_complete(run_dir, args.epochs) and not args.force:
        append_status(status_path, {
            'time': now(),
            'stage': 'train',
            'name': name,
            'status': 'skipped_complete',
            'batch': '',
            'returncode': 0,
            'seconds': 0,
            'epochs_done': results_epochs(run_dir),
            'log': '',
        })
        return True

    batches = [args.batch] + [b for b in args.retry_batch if b != args.batch]
    for batch in batches:
        log_path = args.project / 'logs' / f'train_{name}_b{batch}.log'
        resume_from = run_dir / 'weights' / 'last.pt'
        resume_args = ['--resume-from', str(resume_from)] if resume_from.exists() and not args.force else []
        cmd = [
            sys.executable,
            str(ROOT / 'scripts/run_ablation.py'),
            'train',
            '--only',
            f'^{name}$',
            '--data',
            str(args.data),
            '--project',
            str(args.project),
            '--epochs',
            str(args.epochs),
            '--batch',
            str(batch),
            '--imgsz',
            str(args.imgsz),
            '--device',
            args.device,
            '--workers',
            str(args.workers),
            '--save-period',
            str(args.save_period),
            '--exist-ok',
        ] + resume_args
        if args.quiet_model:
            cmd.append('--quiet-model')
        if args.amp:
            cmd.append('--amp')
        if args.fraction < 1.0:
            cmd += ['--fraction', str(args.fraction)]
        if args.warmup_epochs is not None:
            cmd += ['--warmup-epochs', str(args.warmup_epochs)]
        cmd += ['--patience', str(args.patience)]
        if args.deterministic:
            cmd.append('--deterministic')
        if args.gpu_memory_gb > 0:
            cmd += ['--gpu-memory-gb', str(args.gpu_memory_gb)]
        append_status(status_path, {
            'time': now(),
            'stage': 'train',
            'name': name,
            'status': 'started_resume' if resume_args else 'started',
            'batch': batch,
            'returncode': '',
            'seconds': '',
            'epochs_done': results_epochs(run_dir),
            'log': str(log_path),
        })
        returncode, seconds = run_logged(cmd, log_path)
        status = 'complete' if returncode == 0 and is_complete(run_dir, args.epochs) else 'failed'
        append_status(status_path, {
            'time': now(),
            'stage': 'train',
            'name': name,
            'status': status,
            'batch': batch,
            'returncode': returncode,
            'seconds': f'{seconds:.1f}',
            'epochs_done': results_epochs(run_dir),
            'log': str(log_path),
        })
        if status == 'complete':
            return True
    return False


def plot_summary(args, status_path):
    log_path = args.project / 'logs' / 'plot_summary.log'
    cmd = [
        sys.executable,
        str(ROOT / 'scripts/plot_ablation_results.py'),
        '--project',
        str(args.project),
        '--out',
        str(args.project / 'summary'),
    ]
    returncode, seconds = run_logged(cmd, log_path)
    append_status(status_path, {
        'time': now(),
        'stage': 'summary',
        'name': 'all',
        'status': 'complete' if returncode == 0 else 'failed',
        'batch': '',
        'returncode': returncode,
        'seconds': f'{seconds:.1f}',
        'epochs_done': '',
        'log': str(log_path),
    })
    return returncode == 0


def visualize_one(name, cfg, args, status_path):
    if name == 'baseline':
        return True
    weights = args.project / name / 'weights' / 'best.pt'
    if not weights.exists():
        append_status(status_path, {
            'time': now(),
            'stage': 'visualize',
            'name': name,
            'status': 'skipped_no_weights',
            'batch': '',
            'returncode': '',
            'seconds': '',
            'epochs_done': results_epochs(args.project / name),
            'log': '',
        })
        return False

    device = args.device
    if device.isdigit():
        device = f'cuda:{device}'
    log_path = args.project / 'logs' / f'visualize_{name}.log'
    cmd = [
        sys.executable,
        str(ROOT / 'scripts/visualize_sch.py'),
        '--cfg',
        str(cfg),
        '--weights',
        str(weights),
        '--source',
        str(args.visual_source),
        '--out',
        str(args.project / 'visualizations' / name),
        '--imgsz',
        str(args.imgsz),
        '--device',
        device,
        '--limit',
        str(args.visual_limit),
        '--classes',
        str(args.classes),
    ]
    returncode, seconds = run_logged(cmd, log_path)
    append_status(status_path, {
        'time': now(),
        'stage': 'visualize',
        'name': name,
        'status': 'complete' if returncode == 0 else 'failed',
        'batch': '',
        'returncode': returncode,
        'seconds': f'{seconds:.1f}',
        'epochs_done': results_epochs(args.project / name),
        'log': str(log_path),
    })
    return returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', default='', help='Regex filter for experiment names.')
    parser.add_argument('--data', type=Path, default=ROOT / 'dataset/coco_UAV.yaml')
    parser.add_argument('--project', type=Path, default=ROOT / 'run/full_suite_300e_224')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--retry-batch', type=int, nargs='*', default=[12, 8])
    parser.add_argument('--imgsz', type=int, default=224)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--save-period', type=int, default=-1)
    parser.add_argument('--visual-source', type=Path, default=ROOT / 'dataset/coco_uav/images/val')
    parser.add_argument('--classes', type=Path, default=ROOT / 'dataset/coco_uav/classes.txt')
    parser.add_argument('--visual-limit', type=int, default=20)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--visualize-only', action='store_true')
    parser.add_argument('--no-visualize', action='store_true')
    parser.add_argument('--quiet-model', action='store_true', help='Pass --quiet-model to training subprocesses.')
    parser.add_argument('--amp', action='store_true', help='Pass --amp to training subprocesses.')
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='Pass a training data fraction to subprocesses. Keep 1.0 for full training.')
    parser.add_argument('--warmup-epochs', type=float, default=None,
                        help='Pass a warmup_epochs override to training subprocesses.')
    parser.add_argument('--patience', type=int, default=50,
                        help='Pass early stopping patience to training subprocesses.')
    parser.add_argument('--deterministic', action='store_true',
                        help='Pass --deterministic to training subprocesses. Disabled by default for speed.')
    parser.add_argument('--gpu-memory-gb', type=float, default=0.0,
                        help='Pass a PyTorch CUDA per-process memory cap in GiB to training subprocesses.')
    args = parser.parse_args()

    ensure_configs()
    args.project.mkdir(parents=True, exist_ok=True)
    status_path = args.project / 'suite_status.csv'
    names = selected_names(args.only)

    if not args.summary_only and not args.visualize_only:
        for name in names:
            train_one(name, args, status_path)
            plot_summary(args, status_path)

    plot_summary(args, status_path)

    if not args.no_visualize and not args.summary_only:
        for name in names:
            visualize_one(name, EXPERIMENTS[name], args, status_path)
        plot_summary(args, status_path)

    append_status(status_path, {
        'time': now(),
        'stage': 'suite',
        'name': 'all',
        'status': 'finished',
        'batch': '',
        'returncode': 0,
        'seconds': '',
        'epochs_done': '',
        'log': '',
    })


if __name__ == '__main__':
    main()
