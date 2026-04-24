"""
Weight visualization for HW1 MLP.

Generates:
  - weight_vis.png          : all W1 columns reshaped to 64x64x3
  - weight_class_analysis.png : class avg image + top-activated W1 neurons
  - weight_stats.png        : activation strength & RGB channel bias per class

Usage:
  python visualize.py --ckpt ../outputs/best_model.npz
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_loader import load_dataset, CLASSES, INPUT_DIM, batch_iter
from model import MLP, relu

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'EuroSAT_RGB')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


# ---------------------------------------------------------------------- helpers
def _class_activations(W1, X_test, y_test):
    result = {}
    for cls_name in CLASSES:
        cls_idx = CLASSES.index(cls_name)
        X_cls = X_test[y_test == cls_idx]
        a1 = relu(X_cls @ W1)
        mean_act = a1.mean(axis=0)
        result[cls_name] = (mean_act, np.argsort(mean_act)[::-1])
    return result


def _rgb_bias(W1, top_idx, top_k=6):
    vecs = W1[:, top_idx[:top_k]]
    return np.array([vecs[0::3].mean(), vecs[1::3].mean(), vecs[2::3].mean()])


# ---------------------------------------------------------------- Figure A: W1 overview
def visualize_weights(model, save_path):
    W1 = model.W1
    n_show = min(W1.shape[1], 64)
    cols, rows = 8, (n_show + 7) // 8
    fig = plt.figure(figsize=(cols * 1.5, rows * 1.5))
    gs  = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.05, wspace=0.05)
    for i in range(n_show):
        patch = W1[:, i].reshape(64, 64, 3)
        patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ax.imshow(patch); ax.axis('off')
    fig.suptitle('First-layer weight visualizations (W1 columns)', fontsize=11)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# -------------------------------------------- Figure B: class avg + top neurons
def visualize_class_analysis(model, X_test, y_test, stats,
                              target_classes, top_k, save_path):
    W1 = model.W1
    act_data = _class_activations(W1, X_test, y_test)

    n_cls = len(target_classes)
    n_cols = 1 + top_k
    fig, axes = plt.subplots(n_cls, n_cols, figsize=(n_cols * 1.7, n_cls * 2.0))
    if n_cls == 1:
        axes = axes[np.newaxis, :]

    for row_i, cls_name in enumerate(target_classes):
        cls_idx = CLASSES.index(cls_name)
        X_cls = X_test[y_test == cls_idx]
        mean_act, top_idx = act_data[cls_name]

        avg_img = X_cls.mean(axis=0) * stats['std'] + stats['mean']
        avg_img = np.clip(avg_img.reshape(64, 64, 3) / 255.0, 0, 1)
        axes[row_i, 0].imshow(avg_img)
        axes[row_i, 0].axis('off')
        axes[row_i, 0].set_title(f'{cls_name}\navg image', fontsize=7)

        for col_i, nidx in enumerate(top_idx[:top_k]):
            patch = W1[:, nidx].reshape(64, 64, 3)
            patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
            ax = axes[row_i, col_i + 1]
            ax.imshow(patch); ax.axis('off')
            ax.set_title(f'#{nidx}\n{mean_act[nidx]:.2f}', fontsize=7)

    fig.suptitle('Class avg image  +  top-activated W1 neurons', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# --------------------------------------- Figure C: activation + RGB bias stats
def visualize_stats(model, X_test, y_test, target_classes, top_k, save_path):
    W1 = model.W1
    act_data = _class_activations(W1, X_test, y_test)

    fig, (ax_act, ax_rgb) = plt.subplots(1, 2, figsize=(13, 4))

    # activation strength (all 10 classes)
    top_means = [act_data[c][0][act_data[c][1][:top_k]].mean() for c in CLASSES]
    colors = ['#2ecc71' if c in target_classes else '#95a5a6' for c in CLASSES]
    bars = ax_act.bar(CLASSES, top_means, color=colors, edgecolor='white')
    ax_act.set_xticklabels(CLASSES, rotation=35, ha='right', fontsize=8)
    ax_act.set_ylabel('Mean activation of top neurons')
    ax_act.set_title('First-layer activation strength per class')
    ax_act.grid(axis='y', alpha=0.4)
    for bar, val in zip(bars, top_means):
        ax_act.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    # RGB channel bias (target classes only)
    x = np.arange(len(target_classes))
    width = 0.25
    for ci, (ch, color) in enumerate(zip(['R','G','B'],
                                         ['#e74c3c','#2ecc71','#3498db'])):
        vals = [_rgb_bias(W1, act_data[c][1], top_k)[ci] for c in target_classes]
        ax_rgb.bar(x + ci * width, vals, width, label=ch, color=color,
                   alpha=0.85, edgecolor='white')
    ax_rgb.set_xticks(x + width)
    ax_rgb.set_xticklabels(target_classes, fontsize=9)
    ax_rgb.set_ylabel('Mean weight value')
    ax_rgb.set_title('RGB channel bias of top neurons per class')
    ax_rgb.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax_rgb.legend(title='Channel')
    ax_rgb.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# -------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',    default='../outputs/best_model.npz')
    parser.add_argument('--top_k',   type=int, default=6)
    parser.add_argument('--classes', nargs='+',
                        default=['Forest', 'River', 'Highway'])
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset ...")
    _, _, (X_test, y_test), stats = load_dataset(DATA_DIR)

    ckpt = args.ckpt if os.path.isabs(args.ckpt) else \
           os.path.join(os.path.dirname(__file__), args.ckpt)
    H1, H2 = 512, 256
    model = MLP(INPUT_DIM, H1, H2, len(CLASSES))
    model.load(ckpt)
    print(f"Loaded checkpoint: {ckpt}")

    visualize_weights(model,
        save_path=os.path.join(OUTPUT_DIR, 'weight_vis.png'))

    visualize_class_analysis(model, X_test, y_test, stats,
        target_classes=args.classes, top_k=args.top_k,
        save_path=os.path.join(OUTPUT_DIR, 'weight_class_analysis.png'))

    visualize_stats(model, X_test, y_test,
        target_classes=args.classes, top_k=args.top_k,
        save_path=os.path.join(OUTPUT_DIR, 'weight_stats.png'))


if __name__ == '__main__':
    main()
