"""Tuning sweep for notebook 11: find a noisy-recall config where LSTM beats vanilla RNN."""

import numpy as np
import torch
from torch import nn

from common.dl import set_seed
from common.sequences import make_noisy_recall_task, one_hot

DEVICE = torch.device("cpu")  # deterministic, per section 07 precedent
N_SYMBOLS = 4


class Recurrent(nn.Module):
    def __init__(self, kind, n_in=N_SYMBOLS, n_hidden=32, n_out=N_SYMBOLS):
        super().__init__()
        cell = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind]
        self.rnn = cell(n_in, n_hidden, batch_first=True)
        self.out = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        seq, _ = self.rnn(x)
        return self.out(seq[:, -1])


def fit(kind, seq_len, seed, epochs, lr=0.01, n_hidden=32):
    xs, ys = make_noisy_recall_task(1600, seq_len, n_symbols=N_SYMBOLS, seed=seed)
    X = torch.from_numpy(one_hot(xs, N_SYMBOLS)).to(DEVICE)
    Y = torch.from_numpy(ys).to(DEVICE)
    n_tr = 1200
    set_seed(seed)
    model = Recurrent(kind, n_hidden=n_hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(n_tr, generator=g)
        for i in range(0, n_tr, 50):
            idx = order[i : i + 50]
            opt.zero_grad()
            nn.functional.cross_entropy(model(X[idx]), Y[idx]).backward()
            opt.step()
    with torch.no_grad():
        return (model(X[n_tr:]).argmax(1) == Y[n_tr:]).float().mean().item()


SEEDS = (0, 1, 2, 3, 4)
for seq_len in (20, 25, 30):
    for epochs in (25, 40):
        print(f"\n=== length {seq_len}, epochs {epochs} ===")
        res = {}
        for kind in ("rnn", "lstm"):
            accs = [fit(kind, seq_len, s, epochs) for s in SEEDS]
            res[kind] = np.array(accs)
            print(f"{kind:>4}: {np.round(accs, 3)} mean {np.mean(accs):.3f}")
        margin = (res["lstm"] - res["rnn"]) * 100
        wins = (margin > 0).all()
        print(f"  LSTM beats RNN every seed: {wins}; mean margin {margin.mean():.1f} pts")
