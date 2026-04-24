import numpy as np
import itertools
import json
import os

from model import MLP
from trainer import SGDTrainer
from data_loader import INPUT_DIM


def _subsample(X, y, n, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return X[idx], y[idx]


def _run_trial(cfg, Xtr, ytr, Xv, yv, epochs, ckpt_path, num_classes):
    model = MLP(
        input_dim=INPUT_DIM,
        hidden_dim1=cfg['hidden_dim1'],
        hidden_dim2=cfg['hidden_dim2'],
        num_classes=num_classes,
        activation=cfg['activation'],
        seed=42,
    )
    trainer = SGDTrainer(
        model,
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
        batch_size=cfg.get('batch_size', 256),
        lr_decay=0.9,
        decay_every=5,
    )
    history = trainer.train(Xtr, ytr, Xv, yv, epochs=epochs,
                            checkpoint_path=ckpt_path, verbose=False)
    return max(history['val_acc']), history


def grid_search(X_train, y_train, X_val, y_val,
                param_grid=None, epochs=10,
                num_classes=10, save_dir='search_results',
                train_subset=3000, val_subset=1500):
    """Grid search with optional data subsampling for speed."""
    if param_grid is None:
        param_grid = {
            'lr':           [0.05, 0.01, 0.005],
            'hidden_dim1':  [256, 512],
            'hidden_dim2':  [128, 256],
            'weight_decay': [1e-4, 1e-3],
            'activation':   ['relu', 'tanh'],
        }

    os.makedirs(save_dir, exist_ok=True)
    Xtr, ytr = _subsample(X_train, y_train, train_subset) if train_subset else (X_train, y_train)
    Xv,  yv  = _subsample(X_val,   y_val,   val_subset)   if val_subset   else (X_val,   y_val)

    keys   = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    results = []

    print(f"Grid search: {len(combos)} combinations × {epochs} epochs")
    print(f"  train={len(Xtr)}, val={len(Xv)}\n")

    for i, combo in enumerate(combos):
        cfg = dict(zip(keys, combo))
        print(f"[{i+1}/{len(combos)}] {cfg}", end='  ', flush=True)
        best_acc, history = _run_trial(
            cfg, Xtr, ytr, Xv, yv, epochs,
            ckpt_path=os.path.join(save_dir, f'ckpt_{i}.npz'),
            num_classes=num_classes,
        )
        print(f"best_val_acc={best_acc:.4f}")
        results.append({'config': cfg, 'best_val_acc': best_acc,
                        'history': history, 'ckpt': f'ckpt_{i}.npz'})

    results.sort(key=lambda r: r['best_val_acc'], reverse=True)
    _save_summary(results, save_dir)
    return results


def random_search(X_train, y_train, X_val, y_val,
                  n_trials=15, epochs=10,
                  num_classes=10, seed=0, save_dir='random_search_results',
                  train_subset=3000, val_subset=1500):
    """Random search over hyperparameter space with optional subsampling."""
    rng = np.random.default_rng(seed)
    os.makedirs(save_dir, exist_ok=True)

    Xtr, ytr = _subsample(X_train, y_train, train_subset) if train_subset else (X_train, y_train)
    Xv,  yv  = _subsample(X_val,   y_val,   val_subset)   if val_subset   else (X_val,   y_val)

    results = []
    print(f"Random search: {n_trials} trials × {epochs} epochs")
    print(f"  train={len(Xtr)}, val={len(Xv)}\n")

    for i in range(n_trials):
        cfg = {
            'lr':           float(rng.choice([0.1, 0.05, 0.01, 0.005, 0.001])),
            'hidden_dim1':  int(rng.choice([128, 256, 512, 1024])),
            'hidden_dim2':  int(rng.choice([64, 128, 256, 512])),
            'weight_decay': float(rng.choice([0, 1e-5, 1e-4, 1e-3])),
            'activation':   str(rng.choice(['relu', 'tanh', 'sigmoid'])),
            'batch_size':   int(rng.choice([128, 256, 512])),
        }
        print(f"[{i+1}/{n_trials}] {cfg}", end='  ', flush=True)
        best_acc, history = _run_trial(
            cfg, Xtr, ytr, Xv, yv, epochs,
            ckpt_path=os.path.join(save_dir, f'ckpt_{i}.npz'),
            num_classes=num_classes,
        )
        print(f"best_val_acc={best_acc:.4f}")
        results.append({'config': cfg, 'best_val_acc': best_acc,
                        'history': history, 'ckpt': f'ckpt_{i}.npz'})

    results.sort(key=lambda r: r['best_val_acc'], reverse=True)
    _save_summary(results, save_dir)
    return results


def _save_summary(results, save_dir):
    summary = [{'rank': i+1, 'val_acc': r['best_val_acc'], 'config': r['config']}
               for i, r in enumerate(results)]
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n=== Top-5 configurations ===")
    for r in summary[:5]:
        print(f"  #{r['rank']}  val_acc={r['val_acc']:.4f}  {r['config']}")
