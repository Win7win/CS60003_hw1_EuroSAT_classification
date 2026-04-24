import numpy as np


def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(x.dtype)

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1.0 - np.tanh(x) ** 2

def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)


ACTIVATIONS = {
    'relu':    (relu,    relu_grad),
    'tanh':    (tanh,    tanh_grad),
    'sigmoid': (sigmoid, sigmoid_grad),
}


class MLP:
    """
    3-layer MLP: input -> hidden1 -> hidden2 -> output
    All gradients computed manually (no autograd framework).
    """

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes,
                 activation='relu', seed=42):
        assert activation in ACTIVATIONS, f"Unknown activation: {activation}"
        self.act_fn, self.act_grad = ACTIVATIONS[activation]
        self.activation = activation

        rng = np.random.default_rng(seed)
        # He initialization for ReLU, Xavier for tanh/sigmoid
        scale1 = np.sqrt(2.0 / input_dim)   if activation == 'relu' else np.sqrt(1.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim1) if activation == 'relu' else np.sqrt(1.0 / hidden_dim1)
        scale3 = np.sqrt(1.0 / hidden_dim2)

        self.W1 = rng.standard_normal((input_dim,   hidden_dim1)) * scale1
        self.b1 = np.zeros(hidden_dim1)
        self.W2 = rng.standard_normal((hidden_dim1, hidden_dim2)) * scale2
        self.b2 = np.zeros(hidden_dim2)
        self.W3 = rng.standard_normal((hidden_dim2, num_classes)) * scale3
        self.b3 = np.zeros(num_classes)

        self._cache = {}

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, X):
        """X: (batch, input_dim) -> logits: (batch, num_classes)"""
        z1 = X @ self.W1 + self.b1           # (B, H1)
        a1 = self.act_fn(z1)                 # (B, H1)
        z2 = a1 @ self.W2 + self.b2          # (B, H2)
        a2 = self.act_fn(z2)                 # (B, H2)
        z3 = a2 @ self.W3 + self.b3          # (B, C)

        self._cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3}
        return z3

    # ------------------------------------------------------------------
    # loss: softmax cross-entropy
    # ------------------------------------------------------------------
    @staticmethod
    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def loss(self, logits, y, weight_decay=0.0):
        """Returns (scalar loss, dlogits)."""
        B = len(y)
        probs = self.softmax(logits)
        log_probs = np.log(probs + 1e-12)
        ce = -log_probs[np.arange(B), y].mean()

        l2 = weight_decay / 2 * (
            (self.W1 ** 2).sum() + (self.W2 ** 2).sum() + (self.W3 ** 2).sum()
        )
        total_loss = ce + l2

        # gradient of softmax cross-entropy
        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1
        dlogits /= B
        return total_loss, dlogits

    # ------------------------------------------------------------------
    # backward
    # ------------------------------------------------------------------
    def backward(self, dlogits, weight_decay=0.0):
        """Compute gradients; return dict of param grads."""
        c = self._cache
        X, z1, a1, z2, a2 = c['X'], c['z1'], c['a1'], c['z2'], c['a2']

        # layer 3
        dW3 = a2.T @ dlogits + weight_decay * self.W3
        db3 = dlogits.sum(axis=0)
        da2 = dlogits @ self.W3.T

        # layer 2
        dz2 = da2 * self.act_grad(z2)
        dW2 = a1.T @ dz2 + weight_decay * self.W2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T

        # layer 1
        dz1 = da1 * self.act_grad(z1)
        dW1 = X.T @ dz1 + weight_decay * self.W1
        db1 = dz1.sum(axis=0)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)

    def load(self, path):
        d = np.load(path)
        self.W1, self.b1 = d['W1'], d['b1']
        self.W2, self.b2 = d['W2'], d['b2']
        self.W3, self.b3 = d['W3'], d['b3']
