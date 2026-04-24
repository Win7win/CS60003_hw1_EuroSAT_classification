"""
Training entry point for HW1: 3-layer MLP on EuroSAT.

Usage:
  # Default training (recommended config)
  python train.py

  # Grid search then full training
  python train.py --search grid

  # Random search then full training
  python train.py --search random

  # Custom hyperparameters
  python train.py --lr 0.01 --hidden_dim1 512 --hidden_dim2 256 --epochs 60

  # Evaluate only (skip training)
  python train.py --test --ckpt path/to/best_model.npz
"""
import argparse
import os
import json
import numpy as np

from data_loader import load_dataset, CLASSES, INPUT_DIM
from model import MLP
from trainer import SGDTrainer
from hyperparameter_search import grid_search, random_search

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'EuroSAT_RGB')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def evaluate_accuracy(model, X, y, batch_size=512):
    from data_loader import batch_iter
    correct = sum((model.predict(xb) == yb).sum()
                  for xb, yb in batch_iter(X, y, batch_size, shuffle=False))
    return correct / len(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search',       choices=['grid', 'random', 'none'], default='none')
    parser.add_argument('--epochs',       type=int,   default=60)
    parser.add_argument('--lr',           type=float, default=0.01)
    parser.add_argument('--hidden_dim1',  type=int,   default=512)
    parser.add_argument('--hidden_dim2',  type=int,   default=256)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--activation',   default='relu',
                        choices=['relu', 'tanh', 'sigmoid'])
    parser.add_argument('--batch_size',   type=int,   default=256)
    parser.add_argument('--test',         action='store_true')
    parser.add_argument('--ckpt',         default='best_model.npz')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUTPUT_DIR, args.ckpt)

    print("Loading dataset ...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), stats = load_dataset(DATA_DIR)
    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
    np.save(os.path.join(OUTPUT_DIR, 'norm_stats.npy'), stats)

    # hyperparameter search
    if args.search != 'none' and not args.test:
        search_dir = os.path.join(OUTPUT_DIR, f'{args.search}_search')
        if args.search == 'grid':
            results = grid_search(X_train, y_train, X_val, y_val,
                                  epochs=10, train_subset=3000, val_subset=1500,
                                  save_dir=search_dir)
        else:
            results = random_search(X_train, y_train, X_val, y_val,
                                    n_trials=15, epochs=10,
                                    train_subset=3000, val_subset=1500,
                                    save_dir=search_dir)
        cfg = results[0]['config']
        print(f"\nBest config from search: {cfg}")
    else:
        cfg = {
            'lr': args.lr, 'hidden_dim1': args.hidden_dim1,
            'hidden_dim2': args.hidden_dim2, 'weight_decay': args.weight_decay,
            'activation': args.activation, 'batch_size': args.batch_size,
        }

    model = MLP(INPUT_DIM, cfg['hidden_dim1'], cfg['hidden_dim2'],
                len(CLASSES), cfg['activation'])

    # training
    if not args.test:
        trainer = SGDTrainer(model, lr=cfg['lr'], weight_decay=cfg['weight_decay'],
                             batch_size=cfg.get('batch_size', 256),
                             lr_decay=0.95, decay_every=5)
        print(f"\nTraining: {cfg}")
        history = trainer.train(X_train, y_train, X_val, y_val,
                                epochs=args.epochs, checkpoint_path=ckpt_path)
        with open(os.path.join(OUTPUT_DIR, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        # plot training curves
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        epochs_range = range(1, len(history['train_loss']) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs_range, history['train_loss'], label='Train')
        ax1.plot(epochs_range, history['val_loss'],   label='Val')
        ax1.set_title('Loss'); ax1.legend(); ax1.grid(True)
        ax2.plot(epochs_range, history['train_acc'],  label='Train')
        ax2.plot(epochs_range, history['val_acc'],    label='Val')
        ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
        plt.close()
        print(f"Training curves saved.")

    # evaluate
    print(f"\nLoading checkpoint: {ckpt_path}")
    model.load(ckpt_path)
    test_acc = evaluate_accuracy(model, X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
