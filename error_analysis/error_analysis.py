"""
Error analysis for HW1 MLP on EuroSAT.

Generates:
  - confusion_matrix.png  : per-class confusion matrix
  - error_analysis.png    : randomly sampled misclassified satellite images

Usage:
  python error_analysis.py --ckpt ../outputs/best_model.npz
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import load_dataset, CLASSES, INPUT_DIM, batch_iter
from model import MLP

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'EuroSAT_RGB')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def evaluate(model, X_test, y_test, batch_size=512):
    all_preds = []
    for xb, _ in batch_iter(X_test, y_test, batch_size, shuffle=False):
        all_preds.append(model.predict(xb))
    preds = np.concatenate(all_preds)
    acc = (preds == y_test).mean()
    n = len(CLASSES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_test, preds):
        cm[t, p] += 1
    return acc, cm, preds


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=7,
                    color='white' if cm[i, j] > cm.max() * 0.5 else 'black')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_error_samples(X_test, y_test, preds, stats, n_samples, save_path):
    wrong_idx = np.where(preds != y_test)[0]
    rng = np.random.default_rng(0)
    chosen = rng.choice(wrong_idx, size=min(n_samples, len(wrong_idx)), replace=False)

    cols = 4
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for ax_i, idx in enumerate(chosen):
        img = X_test[idx] * stats['std'] + stats['mean']
        img = np.clip(img.reshape(64, 64, 3) / 255.0, 0, 1)
        axes[ax_i].imshow(img)
        axes[ax_i].set_title(
            f"True: {CLASSES[y_test[idx]]}\nPred: {CLASSES[preds[idx]]}",
            fontsize=7, color='red')
        axes[ax_i].axis('off')

    for ax_i in range(len(chosen), len(axes)):
        axes[ax_i].axis('off')

    fig.suptitle('Misclassified Examples', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def print_per_class_accuracy(cm):
    print("\nPer-class accuracy:")
    print(f"{'Class':<28} {'Correct':>7} {'Total':>7} {'Acc':>7}")
    print("-" * 52)
    for i, cls in enumerate(CLASSES):
        total   = cm[i].sum()
        correct = cm[i, i]
        print(f"{cls:<28} {correct:>7} {total:>7} {correct/total:>7.1%}")
    print(f"\nOverall accuracy: {cm.diagonal().sum() / cm.sum():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',      default='../outputs/best_model.npz')
    parser.add_argument('--n_samples', type=int, default=12)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset ...")
    _, _, (X_test, y_test), stats = load_dataset(DATA_DIR)

    ckpt = args.ckpt if os.path.isabs(args.ckpt) else \
           os.path.join(os.path.dirname(__file__), args.ckpt)
    model = MLP(INPUT_DIM, 512, 256, len(CLASSES))
    model.load(ckpt)
    print(f"Loaded checkpoint: {ckpt}")

    acc, cm, preds = evaluate(model, X_test, y_test)
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print_per_class_accuracy(cm)

    plot_confusion_matrix(cm,
        save_path=os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plot_error_samples(X_test, y_test, preds, stats,
        n_samples=args.n_samples,
        save_path=os.path.join(OUTPUT_DIR, 'error_analysis.png'))


if __name__ == '__main__':
    main()
