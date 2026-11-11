"""Tiny 1D CNN on filtered PPG windows (25 Hz, 60 s = 1500 samples), LOPO."""
import time
import numpy as np
import torch
from torch import nn
from .models import lopo_splits

torch.set_num_threads(2)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(8, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 16, 7, padding=3), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.reg = nn.Linear(16, 1)
        self.cls = nn.Linear(16, 1)

    def forward(self, x):
        h = self.net(x)
        return self.reg(h).squeeze(1), self.cls(h).squeeze(1)


def fit_predict(xtr, ytr, ctr, xte, epochs=15, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = ytr.mean(), ytr.std() + 1e-6
    m = TinyCNN(); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    X = torch.tensor(xtr[:, None]); Y = torch.tensor((ytr - mu) / sd, dtype=torch.float32)
    C = torch.tensor(ctr, dtype=torch.float32)
    pos = max(C.mean().item(), 1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            b = perm[i:i + 64]
            r, c = m(X[b])
            loss = nn.functional.mse_loss(r, Y[b]) + bce(c, C[b])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        out = []
        for xx in (xtr, xte):
            r, c = m(torch.tensor(xx[:, None]))
            out.append((r.numpy() * sd + mu, torch.sigmoid(c).numpy()))
    return out  # [(train_reg, train_prob), (test_reg, test_prob)]


def run_cnn_lopo(x, y, ich, groups, epochs=15, budget_s=180):
    t0 = time.time()
    reg = np.full(len(y), np.nan); prob = np.full(len(y), np.nan); tr_out = []
    for pid, tr, te in lopo_splits(groups):
        (rtr, ptr), (rte, pte) = fit_predict(x[tr], y[tr], ich[tr], x[te], epochs)
        reg[te], prob[te] = rte, pte
        tr_out.append((tr, rtr, ptr))
        if time.time() - t0 > budget_s:
            raise TimeoutError("CNN over time budget")
    return reg, prob, tr_out, time.time() - t0
