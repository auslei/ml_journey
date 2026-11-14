"""Rebuild 05_deep_learning/11_rnn_and_lstm.ipynb for the polish261114 pass.

Keeps the from-scratch gradient-decay demo on delayed recall (cells 0 to 5) and switches the
trained comparison to the adding problem, where the LSTM reliably beats the vanilla RNN.
"""

import json

PATH = "05_deep_learning/11_rnn_and_lstm.ipynb"

md = []  # markdown sources
code = []  # code sources

md.append(r"""# Recurrent networks: RNN, GRU and LSTM

A recurrent network reads a sequence one step at a time and carries a hidden state forward, so the last step can depend on the first. That is also where it breaks. Unrolled over $T$ steps it is a $T$-layer network with the same weights in every layer, so the unstable gradients of notebook 05 come straight back. This notebook measures how a vanilla RNN's gradient dies with distance, then trains PyTorch's `RNN`, `GRU` and `LSTM` on the adding problem and watches the LSTM pull clear.

Order: intuition, maths, cost function, from-scratch version, library version, tuning, what breaks.""")

code.append(
    """from common.dl import set_seed, setup_keras

setup_keras("torch")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from common.sequences import make_adding_task, make_memory_task, one_hot

SEED = 42
set_seed(SEED)
np.set_printoptions(precision=3, suppress=True)
plt.rcParams["figure.figsize"] = (6, 3.5)
device = torch.device("cpu")  # CPU keeps the trained numbers reproducible across machines
print("device:", device)"""
)

md.append(r"""## 1. Intuition

Start with a task that isolates memory: delayed recall. Each sequence shows a symbol at step 0, then blanks, and the label is that first symbol. The model must hold it for $T-1$ steps with nothing else to help, a clean probe for gradient decay. Sequences come from `make_memory_task`, seeded and tested.

## 2. Maths and cost function

The vanilla RNN updates $\mathbf h_t=\tanh(W_x\mathbf x_t+W_h\mathbf h_{t-1}+\mathbf b)$ and reads from the last state. Backprop through time sends the gradient back one step at a time:

$$\frac{\partial L}{\partial \mathbf h_{t-1}}=W_h^\top\,\mathrm{diag}\big(1-\mathbf h_t^2\big)\,\frac{\partial L}{\partial \mathbf h_t}$$

Over $k$ steps that is $k$ copies of the same $W_h^\top$ and the tanh slope, at most 1. Below a largest singular value of 1 the gradient shrinks geometrically; above 1 the weights fight the tanh slope. It is the product from notebook 05 with one matrix repeated.

GRU and LSTM add gates. The LSTM keeps a separate cell state updated by addition, $\mathbf c_t=\mathbf f_t\odot\mathbf c_{t-1}+\mathbf i_t\odot\tilde{\mathbf c}_t$. When $\mathbf f_t$ is near 1 the gradient passes along the cell state almost unchanged.

## 3. From scratch

A NumPy RNN, forward and backward through time. The recurrent matrix is orthogonal times a scale, so the scale is its largest singular value. Measure the gradient norm at every step for scales 0.5, 1.0 and 1.5 over $T=40$.""")

code.append(
    """x_seq, y_seq = make_memory_task(4, seq_len=8, n_symbols=4, seed=SEED)  # 4 is the blank
print("sequences:\\n", x_seq, "\\nlabels (the first symbol):", y_seq)"""
)

md.append(
    r"""At scale 0.5 the gradient at step 1 is $5.5\times10^{-15}$ of the gradient at step 40. At 1.0, where the weights alone would preserve it, the tanh slope still cuts it to $2\times10^{-3}$. At 1.5 it survives (ratio 3.2), because the weights now outgrow the tanh slope. That is the unstable-gradient fork from notebook 05 along the time axis, and gradient clipping is the standard guard for the exploding side."""
)

code.append(
    '''class NumpyRNN:
    """h_t = tanh(Wx x_t + Wh h_{t-1} + b), then a linear read-out from the last hidden state."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, recurrent_scale: float, seed: int = 0):
        g = np.random.default_rng(seed)
        self.Wx = g.normal(0, 1 / np.sqrt(n_in), (n_in, n_hidden))
        # Orthogonal-ish recurrent matrix, rescaled so its largest singular value is recurrent_scale.
        q, _ = np.linalg.qr(g.normal(size=(n_hidden, n_hidden)))
        self.Wh = q * recurrent_scale
        self.b = np.zeros(n_hidden)
        self.Wo = g.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, n_out))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        h = np.zeros((len(x), self.b.size))
        hs = [h]
        for t in range(x.shape[1]):
            h = np.tanh(x[:, t] @ self.Wx + h @ self.Wh + self.b)
            hs.append(h)
        return hs[-1] @ self.Wo, hs

    def grad_norms_over_time(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Norm of dLoss/dh_t for every step t, by backprop through time from the last step."""
        logits, hs = self.forward(x)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        p[np.arange(len(y)), y] -= 1  # softmax cross-entropy gradient
        dh = (p / len(y)) @ self.Wo.T
        norms = [np.linalg.norm(dh)]
        for t in range(x.shape[1], 1, -1):
            dh = (dh * (1 - hs[t] ** 2)) @ self.Wh.T  # one step back: tanh slope, then Wh
            norms.append(np.linalg.norm(dh))
        return np.array(norms[::-1])  # index 0 = the first time step


T = 40
x_np, y_np = make_memory_task(64, T, n_symbols=4, seed=SEED)
x_oh = one_hot(x_np, 5)
for scale in (0.5, 1.0, 1.5):
    rnn = NumpyRNN(5, 32, 4, recurrent_scale=scale, seed=SEED)
    g = rnn.grad_norms_over_time(x_oh, y_np)
    plt.semilogy(range(1, T + 1), g, label=f"Wh scale {scale}")
    print(f"scale {scale}: gradient at step 1 / step {T} = {g[0] / g[-1]:.1e}")
plt.xlabel("time step")
plt.ylabel("norm of dLoss/dh_t")
plt.legend()
plt.show()'''
)

md.append(r"""## 4. Library version

Now PyTorch's `nn.RNN`, `nn.GRU` and `nn.LSTM`, which share one interface. First the same question at default initialisation: how much does the loss feel each input step? The model reads only the last step, so the gradient to input $t$ is what survives $T-t$ steps. We measure it on the adding problem, the task we train on next.""")

code.append(
    '''class AddNet(nn.Module):
    """One recurrent layer over the adding problem, with a scalar read-out from the last step."""

    def __init__(self, kind: str, n_hidden: int = 32) -> None:
        super().__init__()
        cell = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind]
        self.rnn = cell(2, n_hidden, batch_first=True)  # two inputs: the value and its marker
        self.out = nn.Linear(n_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.rnn(x)
        return self.out(seq[:, -1]).squeeze(-1)  # read the last step only


T = 30
x_init, _ = make_adding_task(256, T, seed=SEED)
X_init = torch.from_numpy(x_init).to(device)
y_zero = torch.zeros(256, device=device)
for kind in ("rnn", "gru", "lstm"):
    set_seed(SEED)
    net = AddNet(kind).to(device)
    X_init.requires_grad_(True)
    X_init.grad = None
    nn.functional.mse_loss(net(X_init), y_zero).backward()
    g = X_init.grad.norm(dim=(0, 2)).cpu().numpy()  # how much the loss feels each input step
    plt.semilogy(range(1, T + 1), g, label=kind)
    print(f"{kind:>4}: gradient at step 1 / step {T} = {g[0] / g[-1]:.1e}")
plt.xlabel("time step")
plt.ylabel("norm of dLoss/dx_t")
plt.legend()
plt.show()'''
)

md.append(r"""All three vanish at initialisation: step 1 to step 30 is $1.7\times10^{-7}$ for the RNN, $3.0\times10^{-7}$ for the GRU and $3.2\times10^{-6}$ for the LSTM. The gates give a path the network can learn to use, not a free fix. Whether it finds that path is a training question.

## 5. The adding problem

Blank-padded recall lets a vanilla RNN coast: nothing disturbs the hidden state, so it matches the LSTM. The adding problem (Hochreiter and Schmidhuber 1997) removes that free ride. Each step has two channels, a random value in $[0,1)$ and a marker set to 1 at two positions, one per half. The target is the mean of the two marked values, and every other step is interference. Guessing the mean lands about 15% of predictions within the 0.04 tolerance, the floor.""")

code.append(
    '''x_add, y_add = make_adding_task(2, seq_len=8, seed=SEED)
print("one adding sequence (value, marker):\\n", x_add[0])
print("target (mean of the two marked values):", round(float(y_add[0]), 3))


def fit_adding(kind: str, seq_len: int, seed: int, epochs: int = 30) -> float:
    """Train on the adding problem and return the fraction of test predictions within 0.04."""
    xs, ys = make_adding_task(2000, seq_len, seed=seed)
    X, Y = torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device)
    n_tr = 1600
    set_seed(seed)
    model = AddNet(kind).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(n_tr, generator=g)
        for i in range(0, n_tr, 50):
            idx = order[i : i + 50]
            opt.zero_grad()
            nn.functional.mse_loss(model(X[idx]), Y[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # guard the exploding side
            opt.step()
    with torch.no_grad():
        return ((model(X[n_tr:]) - Y[n_tr:]).abs() < 0.04).float().mean().item()


SEEDS = (0, 1, 2, 3, 4)
for kind in ("rnn", "gru", "lstm"):
    accs = [fit_adding(kind, 30, s) for s in SEEDS]
    print(f"{kind:>4}, length 30, five seeds: {np.round(accs, 3)}, mean {np.mean(accs):.3f}")'''
)

md.append(
    r"""The split is clean on every seed. The RNN sits at 0.13 to 0.18, the predict-the-mean floor, so it never isolates the two marked values across the gap. The LSTM scores 0.97 to 1.00 (mean 0.98) and the GRU 1.00 on all five, beating the RNN by 81 to 87 points per seed, a mean margin of 83. When memory must survive interference, the gated cell state carries it."""
)

md.append(r"""## 6. Tuning

Sequence length is the dial. Sweep the RNN and LSTM over lengths 10, 30 and 50, three seeds each.""")

code.append(
    """for seq_len in (10, 30, 50):
    for kind in ("rnn", "lstm"):
        accs = [fit_adding(kind, seq_len, s) for s in (0, 1, 2)]
        print(f"length {seq_len:>2}, {kind:>4}: {np.round(accs, 3)}, mean {np.mean(accs):.3f}")"""
)

md.append(r"""Even at length 10 the RNN reaches only 0.59 while the LSTM is at 1.00. By length 30 the RNN is at the 0.15 floor and stays there at 50; the LSTM holds above 0.97 throughout. The gap is the vanishing gradient, not a tuning artefact.

## 7. What breaks

The gates cost parameters. Each gate is another weight matrix, so at the same hidden size the LSTM is four times the RNN and the GRU three times.""")

code.append(
    """for kind in ("rnn", "gru", "lstm"):
    counts = [sum(p.numel() for p in AddNet(kind, n_hidden=h).rnn.parameters()) for h in (16, 32, 64)]
    print(f"{kind:>4} parameters at hidden size 16, 32, 64: {counts}")"""
)

md.append(
    r"""At hidden size 32 the RNN has 1,152 recurrent parameters, the GRU 3,456 and the LSTM 4,608. That extra cost buys a model that still works at length 50 where the RNN cannot start. Section 06 runs on much longer sequences, where this gap is the whole game."""
)

# interleave: md0, code0, md1, code1, md2, code2, md3, code3, md4, code4, md5(results),
# md6(tuning), code5(tuning), md7(what breaks results), code6(params), md8(close)
order = [
    ("md", 0),
    ("code", 0),
    ("md", 1),
    ("code", 1),
    ("md", 2),
    ("code", 2),
    ("md", 3),
    ("code", 3),
    ("md", 4),
    ("code", 4),
    ("md", 5),
    ("md", 6),
    ("code", 5),
    ("md", 7),
    ("code", 6),
    ("md", 8),
]

cells = []
for n, (kind, i) in enumerate(order):
    src = (md if kind == "md" else code)[i]
    lines = src.split("\n")
    source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
    cell = {
        "cell_type": "markdown" if kind == "md" else "code",
        "id": f"cell{n:02d}",
        "metadata": {},
        "source": source,
    }
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cells.append(cell)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open(PATH, "w") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")
print(f"wrote {PATH} with {len(cells)} cells")
