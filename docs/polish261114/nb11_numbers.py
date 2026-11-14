"""Authoritative numbers for notebook 11, computed exactly as the notebook cells will.

Deterministic CPU run. The notebook's trained cells mirror fit() and the configs below, so the
printed figures here are what nbmake will reproduce and what the prose quotes.
"""

import numpy as np
import torch
from torch import nn

from common.dl import set_seed
from common.sequences import make_adding_task

DEVICE = torch.device("cpu")
TOL = 0.04


class AddNet(nn.Module):
    def __init__(self, kind, n_hidden=32):
        super().__init__()
        cell = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind]
        self.rnn = cell(2, n_hidden, batch_first=True)
        self.out = nn.Linear(n_hidden, 1)

    def forward(self, x):
        seq, _ = self.rnn(x)
        return self.out(seq[:, -1]).squeeze(-1)


def fit_adding(kind, seq_len, seed, epochs=30):
    xs, ys = make_adding_task(2000, seq_len, seed=seed)
    X, Y = torch.from_numpy(xs).to(DEVICE), torch.from_numpy(ys).to(DEVICE)
    n_tr = 1600
    set_seed(seed)
    model = AddNet(kind).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(n_tr, generator=g)
        for i in range(0, n_tr, 50):
            idx = order[i : i + 50]
            opt.zero_grad()
            nn.functional.mse_loss(model(X[idx]), Y[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    with torch.no_grad():
        pred = model(X[n_tr:])
        return ((pred - Y[n_tr:]).abs() < TOL).float().mean().item()


# --- gradient at init on the adding task ---
print("# gradient at init, adding task length 30")
T = 30
xg, _ = make_adding_task(256, T, seed=42)
Xg = torch.from_numpy(xg).to(DEVICE)
Yg = torch.zeros(256)
for kind in ("rnn", "gru", "lstm"):
    set_seed(42)
    net = AddNet(kind).to(DEVICE)
    Xg.requires_grad_(True)
    Xg.grad = None
    nn.functional.mse_loss(net(Xg), Yg).backward()
    g = Xg.grad.norm(dim=(0, 2)).cpu().numpy()
    print(f"{kind:>4}: step1/step{T} = {g[0] / g[-1]:.1e}")

# --- main comparison: length 30, 30 epochs, 5 seeds ---
print("\n# main comparison length 30, 30 epochs, seeds 0..4")
SEEDS5 = (0, 1, 2, 3, 4)
main = {}
for kind in ("rnn", "gru", "lstm"):
    accs = [fit_adding(kind, 30, s) for s in SEEDS5]
    main[kind] = np.array(accs)
    print(f"{kind:>4}: {np.round(accs, 3)} mean {np.mean(accs):.3f}")
margin = (main["lstm"] - main["rnn"]) * 100
print(f"LSTM-RNN margin per seed: {np.round(margin, 1)}; mean {margin.mean():.1f}")
print(f"LSTM beats RNN every seed: {(margin > 0).all()}")

# --- tuning: length dependence, rnn vs lstm, 3 seeds ---
print("\n# tuning: lengths 10/30/50, rnn vs lstm, seeds 0..2")
SEEDS3 = (0, 1, 2)
for seq_len in (10, 30, 50):
    for kind in ("rnn", "lstm"):
        accs = [fit_adding(kind, seq_len, s) for s in SEEDS3]
        print(f"len {seq_len:>2} {kind:>4}: {np.round(accs, 3)} mean {np.mean(accs):.3f}")

# --- parameter counts ---
print("\n# parameter counts (recurrent layer only)")
for kind in ("rnn", "gru", "lstm"):
    counts = [sum(p.numel() for p in AddNet(kind, n_hidden=h).rnn.parameters()) for h in (16, 32, 64)]
    print(f"{kind:>4} at hidden 16/32/64: {counts}")
