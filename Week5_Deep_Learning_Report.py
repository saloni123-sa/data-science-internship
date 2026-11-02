# Week5_Deep_Learning_Report - Virtual Data Science with Python Internship (Saloni Kumari)
# Run with: python Week5_Deep_Learning_Report.py   (needs numpy, pandas, matplotlib, seaborn, scikit-learn, scipy)
import os; os.makedirs('figures', exist_ok=True)

import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
sns.set_theme(style='whitegrid')

digits = load_digits()
X, y = digits.data / 16.0, digits.target          # scale pixels to [0, 1]
print("Images:", X.shape, "| classes:", np.unique(y), "| missing:", int(np.isnan(X).sum()))
print("Images per class:", np.bincount(y).tolist())

X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.2, stratify=y_tmp, random_state=42)
print("Train / validation / test:", len(X_train), len(X_val), len(X_test))

fig, axes = plt.subplots(2, 10, figsize=(10, 2.4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(digits.images[i], cmap='gray_r'); ax.set_title(str(digits.target[i]), fontsize=9); ax.axis('off')
plt.suptitle('First 20 images in the dataset (with true labels)', y=1.05)
plt.savefig('figures/w5_samples.png', dpi=150, bbox_inches='tight'); plt.close()

class MLP:
    """Fully connected network: 64 -> 64 -> 32 -> 10 with ReLU, softmax, dropout, L2 and Adam (NumPy only)."""
    def __init__(self, sizes=(64, 64, 32, 10), seed=0, dropout=0.0, l2=0.0):
        self.rng = np.random.default_rng(seed)
        self.sizes, self.dropout, self.l2 = sizes, dropout, l2
        self.W = [self.rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]   # He init
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]; self.v = [np.zeros_like(p) for p in self.W + self.b]; self.t = 0

    def forward(self, X, train=False):
        acts, masks, a = [X], [], X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            if i < len(self.W) - 1:
                a = np.maximum(z, 0)                                   # ReLU
                if train and self.dropout > 0:                        # inverted dropout
                    mask = (self.rng.random(a.shape) > self.dropout) / (1 - self.dropout); a = a * mask
                else:
                    mask = None
                masks.append(mask)
            else:
                z = z - z.max(axis=1, keepdims=True); e = np.exp(z); a = e / e.sum(axis=1, keepdims=True)   # softmax
            acts.append(a)
        return acts, masks

    def loss(self, probs, y):
        ce = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
        return ce + 0.5 * self.l2 * sum((W ** 2).sum() for W in self.W)

    def backward(self, acts, masks, y):
        n = len(y); gW, gb = [None] * len(self.W), [None] * len(self.W)
        delta = acts[-1].copy(); delta[np.arange(n), y] -= 1; delta /= n          # d(cross-entropy)/d(logits)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta + self.l2 * self.W[i]; gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = delta @ self.W[i].T
                if masks[i - 1] is not None: delta = delta * masks[i - 1]
                delta = delta * (acts[i] > 0)                                       # ReLU derivative
        return gW, gb

    def step(self, gW, gb, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k, (p, g) in enumerate(zip(self.W + self.b, gW + gb)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g; self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            p -= lr * (self.m[k] / (1 - b1 ** self.t)) / (np.sqrt(self.v[k] / (1 - b2 ** self.t)) + eps)

    def predict_proba(self, X): return self.forward(X)[0][-1]
    def predict(self, X): return self.predict_proba(X).argmax(axis=1)

    def fit(self, Xtr, ytr, Xva, yva, epochs=150, lr=3e-3, batch=64, patience=None, verbose=False):
        hist = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}; best = (np.inf, None, 0)
        for ep in range(epochs):
            idx = self.rng.permutation(len(Xtr))
            for s in range(0, len(Xtr), batch):
                j = idx[s:s + batch]; acts, masks = self.forward(Xtr[j], train=True)
                gW, gb = self.backward(acts, masks, ytr[j]); self.step(gW, gb, lr)
            ptr, pva = self.predict_proba(Xtr), self.predict_proba(Xva)
            hist['train_loss'].append(self.loss(ptr, ytr)); hist['val_loss'].append(self.loss(pva, yva))
            hist['train_acc'].append((ptr.argmax(1) == ytr).mean()); hist['val_acc'].append((pva.argmax(1) == yva).mean())
            if patience:
                if hist['val_loss'][-1] < best[0]:
                    best = (hist['val_loss'][-1], ([W.copy() for W in self.W], [b.copy() for b in self.b]), ep)
                elif ep - best[2] >= patience:
                    self.W, self.b = best[1]; hist['stopped_at'] = best[2] + 1; break
        return hist

net = MLP(seed=1, dropout=0.0, l2=0.0)
xb, yb = X_train[:20], y_train[:20]
acts, masks = net.forward(xb); gW, gb = net.backward(acts, masks, yb)
rng = np.random.default_rng(0); worst = 0
for layer in range(3):
    for _ in range(5):
        i, j = rng.integers(net.W[layer].shape[0]), rng.integers(net.W[layer].shape[1]); h = 1e-5
        old = net.W[layer][i, j]
        net.W[layer][i, j] = old + h; lp = net.loss(net.forward(xb)[0][-1], yb)
        net.W[layer][i, j] = old - h; lm = net.loss(net.forward(xb)[0][-1], yb)
        net.W[layer][i, j] = old
        num, ana = (lp - lm) / (2 * h), gW[layer][i, j]
        worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
print("Largest relative error over 15 checked weights: %.2e" % worst)

t0 = time.time()
plain = MLP(seed=7, dropout=0.0, l2=0.0);  h_plain = plain.fit(X_train, y_train, X_val, y_val, epochs=150)
reg = MLP(seed=7, dropout=0.3, l2=1e-4);   h_reg = reg.fit(X_train, y_train, X_val, y_val, epochs=150, patience=20)
print("Training time: %.1f s" % (time.time() - t0))
print("Plain:       epochs=%d | final train acc %.3f | val acc %.3f | final train loss %.4f | final val loss %.4f | min val loss %.4f at epoch %d" % (
    len(h_plain['val_loss']), h_plain['train_acc'][-1], h_plain['val_acc'][-1], h_plain['train_loss'][-1], h_plain['val_loss'][-1], min(h_plain['val_loss']), int(np.argmin(h_plain['val_loss'])) + 1))
print("Regularised: epochs=%d (early stop, best epoch %s) | final train acc %.3f | val acc %.3f | val loss %.4f" % (
    len(h_reg['val_loss']), h_reg.get('stopped_at', 'n/a'), h_reg['train_acc'][-1], h_reg['val_acc'][-1], h_reg['val_loss'][-1]))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for h, name, c in [(h_plain, 'Plain', '#c0504d'), (h_reg, 'Regularised', '#4f81bd')]:
    ax[0].plot(h['train_loss'], '--', color=c, label=f'{name} - train'); ax[0].plot(h['val_loss'], '-', color=c, label=f'{name} - validation')
    ax[1].plot(h['train_acc'], '--', color=c, label=f'{name} - train'); ax[1].plot(h['val_acc'], '-', color=c, label=f'{name} - validation')
ax[0].set_yscale('log'); ax[0].set_title('Loss (log scale; regularised loss includes the L2 term)'); ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Cross-entropy loss'); ax[0].legend(fontsize=8)
ax[1].set_ylim(0.9, 1.005); ax[1].set_title('Accuracy'); ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Accuracy'); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig('figures/w5_curves.png', dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(9, 3.6)); layers = [64, 64, 32, 10]; names = ['Input\n(64 pixels)', 'Hidden 1\n64 ReLU + dropout', 'Hidden 2\n32 ReLU + dropout', 'Output\n10 softmax']
xs = [0, 1, 2, 3]; show = [10, 10, 8, 10]; pos = []
for x, n, s in zip(xs, layers, show):
    ys = np.linspace(-1, 1, s); pos.append([(x, yy) for yy in ys])
    ax.scatter([x] * s, ys, s=90, color=['#9bbb59', '#4f81bd', '#4f81bd', '#c0504d'][x], zorder=3, edgecolor='k')
    ax.text(x, -1.55, names[x], ha='center', fontsize=9)
    if n > s: ax.text(x, 1.25, f'... ({n} units)', ha='center', fontsize=8)
for a, b in zip(pos[:-1], pos[1:]):
    for (x1, y1) in a:
        for (x2, y2) in b: ax.plot([x1, x2], [y1, y2], color='grey', alpha=.12, lw=.6, zorder=1)
ax.set_xlim(-.4, 3.4); ax.set_ylim(-2.1, 1.6); ax.axis('off'); ax.set_title('Network architecture 64 - 64 - 32 - 10 (only some units drawn)')
plt.tight_layout(); plt.savefig('figures/w5_arch.png', dpi=150); plt.close()
print("Trainable parameters:", sum(W.size for W in reg.W) + sum(b.size for b in reg.b))

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

pred_nn = reg.predict(X_test); pred_plain = plain.predict(X_test)
lr = LogisticRegression(max_iter=3000).fit(X_train, y_train)
sk = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=7).fit(X_train, y_train)
res = pd.DataFrame({'Test accuracy': [accuracy_score(y_test, pred_nn), accuracy_score(y_test, pred_plain), accuracy_score(y_test, sk.predict(X_test)), accuracy_score(y_test, lr.predict(X_test))],
                    'Macro F1': [f1_score(y_test, pred_nn, average='macro'), f1_score(y_test, pred_plain, average='macro'), f1_score(y_test, sk.predict(X_test), average='macro'), f1_score(y_test, lr.predict(X_test), average='macro')]},
                   index=['NumPy MLP (regularised)', 'NumPy MLP (plain)', 'sklearn MLPClassifier', 'Logistic regression']).round(4)
print(res)
print()
print(classification_report(y_test, pred_nn, digits=3))

cm = confusion_matrix(y_test, pred_nn)
wrong = np.where(pred_nn != y_test)[0]
fig = plt.figure(figsize=(11, 4.4)); gs = fig.add_gridspec(2, 8)
axc = fig.add_subplot(gs[:, :4]); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axc); axc.set_title('Confusion matrix (test set)'); axc.set_xlabel('Predicted digit'); axc.set_ylabel('True digit')
for k, i in enumerate(wrong[:8]):
    a = fig.add_subplot(gs[k // 4, 4 + k % 4]); a.imshow(X_test[i].reshape(8, 8), cmap='gray_r'); a.set_title(f'true {y_test[i]}\npred {pred_nn[i]}', fontsize=8); a.axis('off')
plt.suptitle('Errors of the final network', y=1.02); plt.tight_layout(); plt.savefig('figures/w5_eval.png', dpi=150, bbox_inches='tight'); plt.close()
print("Misclassified test images:", len(wrong), "of", len(y_test))
print("Pairs (true -> predicted):", [(int(y_test[i]), int(pred_nn[i])) for i in wrong])
