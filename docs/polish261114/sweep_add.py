"""Sweep the adding problem: LSTM vs vanilla RNN. Regression, accuracy = |err| < 0.04."""

import numpy as np
import torch
from torch import nn

from common.dl import set_seed

DEVICE = torch.device("cpu")
TOL = 0.04


def make_adding(n_samples, seq_len, seed):
    rng = np.random.default_rng(seed)
    values = rng.random((n_samples, seq_len), dtype=np.float32)
    markers = np.zeros((n_samples, seq_len), dtype=np.float32)
    p1 = rng.integers(0, seq_len // 2, size=n_samples)
    p2 = rng.integers(seq_len // 2, seq_len, size=n_samples)
    rows = np.arange(n_samples)
    markers[rows, p1] = 1.0
    markers[rows, p2] = 1.0
    x = np.stack([values, markers], axis=-1)  # (n, T, 2)
    y = (values[rows, p1] + values[rows, p2]) / 2.0  # scaled to [0,1]
    return x.astype(np.float32), y.astype(np.float32)


class Reg(nn.Module):
    def __init__(self, kind, n_hidden=32):
        super().__init__()
        cell = {"rnn": nn.RNN, "lstm": nn.LSTM}[kind]
        self.rnn = cell(2, n_hidden, batch_first=True)
        self.out = nn.Linear(n_hidden, 1)

    def forward(self, x):
        seq, _ = self.rnn(x)
        return self.out(seq[:, -1]).squeeze(-1)


def fit(kind, seq_len, seed, epochs, lr=0.01, clip=1.0):
    xs, ys = make_adding(2000, seq_len, seed)
    X, Y = torch.from_numpy(xs).to(DEVICE), torch.from_numpy(ys).to(DEVICE)
    n_tr = 1600
    set_seed(seed)
    model = Reg(kind).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(n_tr, generator=g)
        for i in range(0, n_tr, 50):
            idx = order[i : i + 50]
            opt.zero_grad()
            nn.functional.mse_loss(model(X[idx]), Y[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
    with torch.no_grad():
        pred = model(X[n_tr:])
        return ((pred - Y[n_tr:]).abs() < TOL).float().mean().item()


SEEDS = (0, 1, 2, 3, 4)
for seq_len in (20, 30, 40):
    for epochs in (30, 50):
        print(f"\n=== adding, length {seq_len}, epochs {epochs} ===")
        res = {}
        for kind in ("rnn", "lstm"):
            accs = [fit(kind, seq_len, s, epochs) for s in SEEDS]
            res[kind] = np.array(accs)
            print(f"{kind:>4}: {np.round(accs, 3)} mean {np.mean(accs):.3f}")
        margin = (res["lstm"] - res["rnn"]) * 100
        print(f"  every seed: {(margin > 0).all()}; mean margin {margin.mean():.1f} pts")
