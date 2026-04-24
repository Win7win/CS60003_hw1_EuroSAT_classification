import numpy as np
from data_loader import batch_iter


class SGDTrainer:
    """
    Trains an MLP with:
      - mini-batch SGD
      - step-based learning rate decay
      - L2 weight decay
      - best-model checkpointing based on validation accuracy
    """

    def __init__(self, model, lr=0.01, lr_decay=0.95, decay_every=5,
                 weight_decay=1e-4, batch_size=256):
        self.model        = model
        self.lr           = lr
        self.lr_decay     = lr_decay
        self.decay_every  = decay_every
        self.weight_decay = weight_decay
        self.batch_size   = batch_size

    def _step(self, X_batch, y_batch):
        logits = self.model.forward(X_batch)
        loss, dlogits = self.model.loss(logits, y_batch, self.weight_decay)
        grads = self.model.backward(dlogits, self.weight_decay)

        self.model.W1 -= self.lr * grads['W1']
        self.model.b1 -= self.lr * grads['b1']
        self.model.W2 -= self.lr * grads['W2']
        self.model.b2 -= self.lr * grads['b2']
        self.model.W3 -= self.lr * grads['W3']
        self.model.b3 -= self.lr * grads['b3']
        return loss

    def _eval_loss(self, X, y, batch_size=512):
        losses, counts = [], []
        for xb, yb in batch_iter(X, y, batch_size, shuffle=False):
            logits = self.model.forward(xb)
            loss, _ = self.model.loss(logits, yb, self.weight_decay)
            losses.append(loss * len(yb))
            counts.append(len(yb))
        return sum(losses) / sum(counts)

    def _accuracy(self, X, y, batch_size=512):
        correct = 0
        for xb, yb in batch_iter(X, y, batch_size, shuffle=False):
            preds = self.model.predict(xb)
            correct += (preds == yb).sum()
        return correct / len(y)

    def train(self, X_train, y_train, X_val, y_val,
              epochs=50, checkpoint_path='best_model.npz', verbose=True):
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc':  [], 'val_acc':  [],
        }
        best_val_acc = -1.0
        current_lr   = self.lr

        for epoch in range(1, epochs + 1):
            # LR decay
            if epoch > 1 and (epoch - 1) % self.decay_every == 0:
                current_lr *= self.lr_decay
                self.lr = current_lr

            # one epoch of SGD
            epoch_losses = []
            for xb, yb in batch_iter(X_train, y_train, self.batch_size):
                loss = self._step(xb, yb)
                epoch_losses.append(loss)

            train_loss = float(np.mean(epoch_losses))
            val_loss   = float(self._eval_loss(X_val, y_val))
            train_acc  = float(self._accuracy(X_train, y_train))
            val_acc    = float(self._accuracy(X_val,   y_val))

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.model.save(checkpoint_path)

            if verbose:
                print(f"Epoch {epoch:3d}/{epochs} | lr={current_lr:.5f} | "
                      f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
                      f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
                      + (" *" if val_acc == best_val_acc else ""))

        return history
