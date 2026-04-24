"""
Test script for HW1: load trained weights and evaluate on the test set.

Usage:
  python test.py --ckpt ../outputs/best_model.npz
"""
import argparse
import os
import sys
import numpy as np

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
    return acc, cm


def print_confusion_matrix(cm):
    col_w = 8
    header = f"{'':22}" + "".join(f"{c[:col_w]:>{col_w}}" for c in CLASSES)
    print(header)
    print("-" * len(header))
    for i, cls in enumerate(CLASSES):
        row = f"{cls:<22}" + "".join(f"{cm[i, j]:>{col_w}}" for j in range(len(CLASSES)))
        print(row)


def print_per_class_accuracy(cm):
    print(f"\n{'Class':<28} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print("-" * 56)
    for i, cls in enumerate(CLASSES):
        total, correct = cm[i].sum(), cm[i, i]
        print(f"{cls:<28} {correct:>8} {total:>8} {correct/total:>8.2%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',        default='../outputs/best_model.npz',
                        help='Path to .npz checkpoint')
    parser.add_argument('--data_dir',    default=None,
                        help='Path to EuroSAT_RGB folder (default: ../data/EuroSAT_RGB)')
    parser.add_argument('--hidden_dim1', type=int, default=512)
    parser.add_argument('--hidden_dim2', type=int, default=256)
    parser.add_argument('--activation',  default='relu',
                        choices=['relu', 'tanh', 'sigmoid'])
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir else DATA_DIR
    print("Loading dataset ...")
    _, _, (X_test, y_test), _ = load_dataset(data_dir)
    print(f"  test set: {len(X_test)} samples")

    ckpt = args.ckpt if os.path.isabs(args.ckpt) else \
           os.path.normpath(os.path.join(os.path.dirname(__file__), args.ckpt))
    print(f"Loading checkpoint: {ckpt}")
    model = MLP(INPUT_DIM, args.hidden_dim1, args.hidden_dim2,
                len(CLASSES), args.activation)
    model.load(ckpt)

    acc, cm = evaluate(model, X_test, y_test)

    print(f"\nTest Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print("\nConfusion Matrix:")
    print_confusion_matrix(cm)
    print_per_class_accuracy(cm)


if __name__ == '__main__':
    main()
