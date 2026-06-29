import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import RTDETR


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
DEFAULT_CLASS_NAMES = [
    'Casing_base', 'Insulator', 'Insulator_base', 'Isoelectric_line',
    'Load_bearing_cable_base', 'Locator_bracing_base', 'Locator_bracing_base_ear',
    'Locator_clamp', 'Locator_hook', 'Locator_ring', 'Locator_tube_connector',
    'Rectangular_locator', 'Rotary_double_ear', 'Sleeve_double_ear', 'Sleeve_screw',
    'Windproof_wire', 'Windproof_wire_ring',
]


def collect_images(source, limit):
    source = Path(source)
    if source.is_file():
        paths = [source]
    else:
        paths = sorted(p for p in source.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
    return paths[:limit] if limit else paths


def load_class_names(path):
    if path and Path(path).exists():
        names = [line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
        return names or DEFAULT_CLASS_NAMES
    return DEFAULT_CLASS_NAMES


def read_image(path, imgsz):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return rgb, resized, tensor


def infer_label_path(image_path):
    text = str(image_path)
    for sep in ('\\', '/'):
        marker = f'{sep}images{sep}'
        if marker in text:
            return Path(text.replace(marker, f'{sep}labels{sep}')).with_suffix('.txt')
    return image_path.with_suffix('.txt')


def load_yolo_boxes(image_path, image_shape, class_names):
    h, w = image_shape[:2]
    label_path = infer_label_path(Path(image_path))
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = parts
        cls = int(float(cls))
        cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        name = class_names[cls] if cls < len(class_names) else str(cls)
        boxes.append((x1, y1, x2, y2, name))
    return boxes


def draw_boxes(ax, boxes, color='#ff2a2a', label_prefix='GT', linewidth=2.0):
    for x1, y1, x2, y2, name in boxes:
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=linewidth)
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 4), f'{label_prefix}:{name}', color='white', fontsize=8,
                bbox=dict(facecolor=color, edgecolor='none', alpha=0.85, pad=1.5))


def structure_heatmap(uhr, image_shape):
    feat = uhr.last_structure_feat[0].float().detach().cpu()
    heat = feat.abs().mean(0).numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    return cv2.resize(heat, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC), feat.shape[-2:]


def maybe_load_model(cfg, weights, device):
    try:
        model = RTDETR(cfg)
        if weights:
            model.load(weights)
    except Exception:
        if not weights:
            raise
        model = RTDETR(weights)
    return model.model.to(device).eval()


def plot_route(image, scores, selected, candidate_shapes, save_path):
    labels = [f'S{i + 1}' for i in range(len(scores))]
    colors = ['#b8c0cc'] * len(scores)
    for idx in selected:
        colors[int(idx)] = '#d1495b'

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title('Input')

    axes[1].bar(labels, scores, color=colors)
    axes[1].set_title(f'DTSR scores, TopK={list(map(int, selected))}')
    axes[1].set_ylabel('score')
    if candidate_shapes:
        for i, shape in enumerate(candidate_shapes):
            axes[1].text(i, scores[i], f'{shape[0]}x{shape[1]}', ha='center', va='bottom', fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_semantic(response, class_names, save_path, topn=17):
    order = np.argsort(response)[::-1][:topn]
    labels = [class_names[i] if i < len(class_names) else f'C{i}' for i in order][::-1]
    values = response[order][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(labels))), dpi=160)
    ax.barh(labels, values, color='#287c8e')
    ax.set_title('CLIP semantic response')
    ax.set_xlabel('cosine response')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def normalized_similarity(x):
    x = F.normalize(x, dim=-1)
    return x @ x.transpose(-1, -2)


def plot_uhr(uhr, save_path, max_queries=80):
    if not hasattr(uhr, 'last_queries_before'):
        return False

    q0 = uhr.last_queries_before[0, :max_queries].detach().cpu()
    q1 = uhr.last_queries_after[0, :max_queries].detach().cpu()
    b0 = uhr.last_boxes_before[0].sigmoid().detach().cpu().numpy()
    b1 = uhr.last_boxes_after[0].sigmoid().detach().cpu().numpy()

    s0 = normalized_similarity(q0).numpy()
    s1 = normalized_similarity(q1).numpy()
    delta = np.linalg.norm(q1.numpy() - q0.numpy(), axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=160)
    axes[0, 0].scatter(b0[:, 0], b0[:, 1], s=8, c='#8293a7', label='before')
    axes[0, 0].scatter(b1[:, 0], b1[:, 1], s=8, c='#d1495b', label='after')
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(1, 0)
    axes[0, 0].set_title('Reference box centers')
    axes[0, 0].legend(loc='upper right')

    axes[0, 1].hist(delta, bins=24, color='#287c8e')
    axes[0, 1].set_title('Query update norm')

    im0 = axes[1, 0].imshow(s0, vmin=-1, vmax=1, cmap='coolwarm')
    axes[1, 0].set_title('Query similarity before UHR')
    axes[1, 0].axis('off')
    im1 = axes[1, 1].imshow(s1, vmin=-1, vmax=1, cmap='coolwarm')
    axes[1, 1].set_title('Query similarity after UHR')
    axes[1, 1].axis('off')
    fig.colorbar(im1, ax=axes[1, :].ravel().tolist(), shrink=0.72)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return True


def plot_hypergraph_feature(uhr, image, save_path, max_queries=80):
    if not hasattr(uhr, 'last_structure_feat') or not hasattr(uhr, 'last_hyper_message'):
        return False

    feat = uhr.last_structure_feat[0].float().detach().cpu()
    heat = feat.abs().mean(0).numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    heat = cv2.resize(heat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    q0 = uhr.last_queries_before[0, :max_queries].detach().cpu()
    q1 = uhr.last_queries_after[0, :max_queries].detach().cpu()
    b0 = uhr.last_boxes_before[0, :max_queries].sigmoid().detach().cpu().numpy()
    b1 = uhr.last_boxes_after[0, :max_queries].sigmoid().detach().cpu().numpy()
    msg = uhr.last_hyper_message[0, :max_queries].detach().cpu()
    msg_norm = torch.linalg.norm(msg, dim=-1).numpy()
    query_delta = torch.linalg.norm(q1 - q0, dim=-1).numpy()

    context_names = ['query', 'category', 'bbox', 'structure', 'message']
    context_values = [
        torch.linalg.norm(uhr.last_queries_before[0], dim=-1).mean().item(),
        torch.linalg.norm(uhr.last_category_context[0], dim=-1).mean().item(),
        torch.linalg.norm(uhr.last_bbox_context[0], dim=-1).mean().item(),
        torch.linalg.norm(uhr.last_structure_context[0], dim=-1).mean().item(),
        torch.linalg.norm(uhr.last_hyper_message[0], dim=-1).mean().item(),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=160)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Input')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(image)
    axes[0, 1].imshow(heat, cmap='magma', alpha=0.48)
    axes[0, 1].set_title(f'UHR structure feature map ({feat.shape[1]}x{feat.shape[2]})')
    axes[0, 1].axis('off')

    sc = axes[1, 0].scatter(b0[:, 0], b0[:, 1], c=msg_norm, s=20, cmap='viridis')
    axes[1, 0].scatter(b1[:, 0], b1[:, 1], s=8, c='#d1495b', label='after')
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(1, 0)
    axes[1, 0].set_title('Query centers colored by hyper-message norm')
    axes[1, 0].legend(loc='upper right')
    fig.colorbar(sc, ax=axes[1, 0], shrink=0.82)

    x = np.arange(len(context_names))
    axes[1, 1].bar(x, context_values, color=['#8293a7', '#287c8e', '#d99a2b', '#4e9f50', '#d1495b'])
    axes[1, 1].set_xticks(x, context_names, rotation=20)
    axes[1, 1].set_title(f'UHR context norm, mean query update={query_delta.mean():.3f}')
    axes[1, 1].set_ylabel('L2 norm')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return True


def plot_uhr_explainable(uhr, image, boxes, save_path, max_queries=80, top_arrows=35):
    if not hasattr(uhr, 'last_structure_feat') or not hasattr(uhr, 'last_hyper_message'):
        return False

    h, w = image.shape[:2]
    heat, heat_shape = structure_heatmap(uhr, image.shape)
    q0 = uhr.last_queries_before[0, :max_queries].detach().cpu()
    q1 = uhr.last_queries_after[0, :max_queries].detach().cpu()
    b0 = uhr.last_boxes_before[0, :max_queries].sigmoid().detach().cpu().numpy()
    b1 = uhr.last_boxes_after[0, :max_queries].sigmoid().detach().cpu().numpy()
    msg = uhr.last_hyper_message[0, :max_queries].detach().cpu()
    msg_norm = torch.linalg.norm(msg, dim=-1).numpy()
    query_delta = torch.linalg.norm(q1 - q0, dim=-1).numpy()

    order = np.argsort(msg_norm)[::-1][:min(top_arrows, len(msg_norm))]
    norm_min, norm_max = float(msg_norm.min()), float(msg_norm.max())

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=160)

    axes[0, 0].imshow(image)
    draw_boxes(axes[0, 0], boxes)
    axes[0, 0].set_title('Input + GT box')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(image)
    axes[0, 1].imshow(heat, cmap='magma', alpha=0.48)
    draw_boxes(axes[0, 1], boxes)
    axes[0, 1].set_title(f'UHR structure heatmap ({heat_shape[0]}x{heat_shape[1]}) + GT')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(image)
    cmap = plt.get_cmap('viridis')
    for idx in order:
        x0, y0 = b0[idx, 0] * w, b0[idx, 1] * h
        x1, y1 = b1[idx, 0] * w, b1[idx, 1] * h
        color = cmap((msg_norm[idx] - norm_min) / (norm_max - norm_min + 1e-6))
        axes[1, 0].annotate('', xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.3, alpha=0.85))
        axes[1, 0].scatter([x1], [y1], s=10, c=[color], edgecolors='white', linewidths=0.25)
    draw_boxes(axes[1, 0], boxes)
    axes[1, 0].set_title(f'Top query updates on image, mean update={query_delta.mean():.3f}')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(image)
    axes[1, 1].imshow(heat, cmap='magma', alpha=0.50)
    draw_boxes(axes[1, 1], boxes, linewidth=2.5)
    if boxes:
        x1, y1, x2, y2, _ = boxes[0]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        pad = max(40, 4.0 * max(x2 - x1, y2 - y1))
        axes[1, 1].set_xlim(max(0, cx - pad), min(w, cx + pad))
        axes[1, 1].set_ylim(min(h, cy + pad), max(0, cy - pad))
        axes[1, 1].set_title('Target zoom: GT + UHR heatmap')
    else:
        axes[1, 1].set_title('GT unavailable: UHR heatmap')
    axes[1, 1].axis('off')

    scalar = plt.cm.ScalarMappable(cmap='viridis')
    scalar.set_array(msg_norm)
    cbar = fig.colorbar(scalar, ax=axes[1, 0], fraction=0.046, pad=0.02)
    cbar.set_label('hyper-message norm')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return True


def plot_route_frequency(rows, out_dir):
    if not rows:
        return
    m = len(rows[0]['scores'])
    counts = np.zeros(m, dtype=np.int64)
    score_stack = []
    for row in rows:
        score_stack.append(row['scores'])
        for idx in row['selected']:
            counts[int(idx)] += 1
    scores = np.asarray(score_stack)
    labels = [f'S{i + 1}' for i in range(m)]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.bar(labels, counts, color='#d1495b')
    ax.set_title('DTSR selected-scale frequency')
    ax.set_ylabel('count')
    fig.tight_layout()
    fig.savefig(out_dir / 'route_frequency.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.errorbar(labels, scores.mean(0), yerr=scores.std(0), fmt='o-', color='#287c8e', capsize=4)
    ax.set_title('DTSR score mean/std')
    ax.set_ylabel('score')
    fig.tight_layout()
    fig.savefig(out_dir / 'route_score_distribution.png')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='ultralytics/cfg/models/rt-detr/sch-rtdetr-mamba.yaml')
    parser.add_argument('--weights', default='')
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', default='run/visualize_sch')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--limit', type=int, default=16)
    parser.add_argument('--classes', default='dataset/coco_uav/classes.txt')
    parser.add_argument('--uhr-queries', type=int, default=80)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names(args.classes)
    paths = collect_images(args.source, args.limit)
    if not paths:
        raise FileNotFoundError(f'No images found under {args.source}')

    device = torch.device(args.device)
    net = maybe_load_model(args.cfg, args.weights or None, device)
    backbone = net.model[0]
    decoder = net.model[-1]

    rows = []
    for path in paths:
        original, resized, tensor = read_image(path, args.imgsz)
        tensor = tensor.to(device)
        with torch.no_grad():
            _ = net(tensor)

        if not hasattr(backbone, 'last_route_scores'):
            raise RuntimeError('Backbone does not expose SCH route scores. Use an SCH-MDETR config or weights.')

        stem = path.stem
        scores = backbone.last_route_scores[0].float().cpu().numpy()
        selected = backbone.last_route_indices.cpu().numpy().astype(int).tolist()
        candidate_shapes = getattr(backbone, 'last_candidate_shapes', [])
        response = backbone.last_semantic_response[0].float().cpu().numpy()
        top_semantic = np.argsort(response)[::-1][:5].astype(int).tolist()
        boxes = load_yolo_boxes(path, resized.shape, class_names)

        plot_route(resized, scores, selected, candidate_shapes, out_dir / f'{stem}_route.png')
        plot_semantic(response, class_names, out_dir / f'{stem}_semantic.png')
        if hasattr(decoder, 'uhr'):
            plot_uhr(decoder.uhr, out_dir / f'{stem}_uhr.png', max_queries=args.uhr_queries)
            plot_hypergraph_feature(decoder.uhr, resized, out_dir / f'{stem}_hypergraph_feature.png',
                                    max_queries=args.uhr_queries)
            plot_uhr_explainable(decoder.uhr, resized, boxes, out_dir / f'{stem}_uhr_explainable.png',
                                 max_queries=args.uhr_queries)

        rows.append({
            'image': str(path),
            'scores': scores.tolist(),
            'selected': selected,
            'top_semantic': top_semantic,
        })

    plot_route_frequency(rows, out_dir)
    with (out_dir / 'route_records.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'scores', 'selected', 'top_semantic'])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'image': row['image'],
                'scores': json.dumps(row['scores']),
                'selected': json.dumps(row['selected']),
                'top_semantic': json.dumps(row['top_semantic']),
            })
    print(f'Wrote SCH visualizations to {out_dir.resolve()}')


if __name__ == '__main__':
    main()
