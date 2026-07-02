import argparse
import contextlib
import io
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import RTDETR


def make_batch(batch_size, imgsz, device):
    img = torch.rand(batch_size, 3, imgsz, imgsz, device=device)
    batch_idx = []
    cls = []
    boxes = []
    for i in range(batch_size):
        batch_idx.extend([i, i])
        cls.extend([[0.], [0.]])
        boxes.extend([[0.50, 0.50, 0.20, 0.20], [0.30, 0.35, 0.12, 0.16]])
    return {
        'img': img,
        'batch_idx': torch.tensor(batch_idx, dtype=torch.long, device=device),
        'cls': torch.tensor(cls, dtype=torch.float32, device=device),
        'bboxes': torch.tensor(boxes, dtype=torch.float32, device=device),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='ultralytics/cfg/models/rt-detr/ablations/sch_full_k3.yaml')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 2, 3, 4, 6, 8, 10, 12, 16])
    parser.add_argument('--device', default='0')
    parser.add_argument('--gpu-memory-gb', type=float, default=0.0)
    parser.add_argument('--amp', action='store_true', help='Probe with CUDA automatic mixed precision.')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if args.device.isdigit() else args.device)
    if args.gpu_memory_gb and device.type == 'cuda':
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        fraction = max(0.01, min(float(args.gpu_memory_gb) / total_gb, 1.0))
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        print(f'{device} per-process memory cap: {args.gpu_memory_gb:.2f} GiB ({fraction:.3f} of {total_gb:.2f} GiB)')
    with contextlib.redirect_stdout(io.StringIO()):
        model = RTDETR(args.cfg, verbose=False)
    net = model.model.to(device).train()

    for batch_size in args.batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            batch = make_batch(batch_size, args.imgsz, device)
            with torch.amp.autocast(device_type='cuda', enabled=args.amp and device.type == 'cuda'):
                loss, _ = net.loss(batch)
            loss.backward()
            peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
            print(f'batch={batch_size} ok peak_allocated={peak:.2f}GiB peak_reserved={reserved:.2f}GiB loss={float(loss.detach()):.3f}')
            net.zero_grad(set_to_none=True)
            del batch, loss
        except torch.cuda.OutOfMemoryError:
            peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
            print(f'batch={batch_size} oom peak_allocated={peak:.2f}GiB peak_reserved={reserved:.2f}GiB')
            break


if __name__ == '__main__':
    main()
