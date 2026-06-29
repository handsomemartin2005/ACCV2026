import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


METRIC_KEYS = [
    'metrics/precision(B)',
    'metrics/recall(B)',
    'metrics/mAP50(B)',
    'metrics/mAP50-95(B)',
]

LOSS_KEYS = [
    'train/giou_loss',
    'train/cls_loss',
    'train/l1_loss',
    'val/giou_loss',
    'val/cls_loss',
    'val/l1_loss',
]


def normalize_columns(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def find_results(project, names):
    project = Path(project)
    runs = [project / n for n in names] if names else sorted(p for p in project.iterdir() if p.is_dir())
    result_files = []
    for run in runs:
        csv_path = run / 'results.csv'
        if csv_path.exists():
            result_files.append((run.name, csv_path))
    return result_files


def plot_curves(frames, keys, out_path, title):
    if not frames:
        return
    cols = 2
    rows = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4 * rows), dpi=160)
    axes = axes.reshape(-1)
    for ax, key in zip(axes, keys):
        for name, df in frames:
            if key in df.columns:
                ax.plot(df[key], label=name, linewidth=1.8)
        ax.set_title(key)
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.25)
    for ax in axes[len(keys):]:
        ax.axis('off')
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_summary(summary, out_path):
    if summary.empty:
        return
    key = 'best_metrics/mAP50-95(B)'
    if key not in summary.columns:
        return
    ordered = summary.sort_values(key)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(ordered))), dpi=160)
    ax.barh(ordered['name'], ordered[key], color='#287c8e')
    ax.set_xlabel('best mAP50-95')
    ax.set_title('Ablation summary')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='run/ablation')
    parser.add_argument('--names', nargs='*', default=[])
    parser.add_argument('--out', default='run/ablation/summary')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    summary_rows = []
    for name, csv_path in find_results(args.project, args.names):
        df = normalize_columns(pd.read_csv(csv_path))
        frames.append((name, df))
        row = {'name': name, 'epochs': len(df)}
        for key in METRIC_KEYS:
            if key in df.columns:
                row[f'best_{key}'] = df[key].max()
                row[f'last_{key}'] = df[key].iloc[-1]
        for key in LOSS_KEYS:
            if key in df.columns:
                row[f'last_{key}'] = df[key].iloc[-1]
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / 'ablation_summary.csv', index=False)
    plot_curves(frames, METRIC_KEYS, out_dir / 'metric_curves.png', 'Ablation metrics')
    plot_curves(frames, LOSS_KEYS, out_dir / 'loss_curves.png', 'Ablation losses')
    plot_summary(summary, out_dir / 'map50_95_bar.png')
    print(f'Wrote ablation plots to {out_dir.resolve()}')


if __name__ == '__main__':
    main()
