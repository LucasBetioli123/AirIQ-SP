"""MLP pequena com saída tanh, em PyTorch, para a competição de modelos.

Ideia testada: com a ativação tanh no neurônio de saída e o alvo normalizado para
[-0.7, +0.7], a rede trabalha na região quase linear da tanh. tanh'(x) = 1 - tanh(x)^2,
e para |tanh| <= 0.7 a derivada fica >= 0.51, então o gradiente não morre perto dos
extremos da faixa de treino. A saída também fica limitada à faixa vista no treino, o que
evita a extrapolação explosiva da MLP de saída linear. Em troca, a rede não consegue
prever acima do máximo do treino.

Arquitetura: entrada -> 64 (tanh) -> 32 (tanh) -> 1 (tanh).
Treino: MSE + weight decay (L2), dropout opcional nas camadas ocultas, Adam,
mini-batches de 128 e early stopping numa fração de validação (pesos da melhor época).
Entradas: imputação pela mediana (com indicadores de ausência) + QuantileTransformer
para normal, robusto às caudas longas de fogo e NO2.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from torch import nn


class TanhMLP(nn.Module):
    def __init__(self, n_in: int, hidden=(64, 32), dropout: float = 0.0):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()] + ([nn.Dropout(dropout)] if dropout > 0 else [])
            prev = h
        layers += [nn.Linear(prev, 1), nn.Tanh()]  # saída tanh
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):  # Xavier: adequada para tanh
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchTanhRegressor:
    """Interface sklearn-like (fit/predict) em volta da TanhMLP."""

    def __init__(self, hidden=(64, 32), out_range: float = 0.7, weight_decay: float = 1e-3, lr: float = 1e-3,
                 batch: int = 128, epochs: int = 400, patience: int = 25, val_frac: float = 0.15, seed: int = 42,
                 dropout: float = 0.0):
        self.hidden, self.out_range, self.weight_decay, self.lr = hidden, out_range, weight_decay, lr
        self.dropout = dropout
        self.batch, self.epochs, self.patience, self.val_frac, self.seed = batch, epochs, patience, val_frac, seed

    # alvo -> [-out_range, +out_range] (min-max do treino) e volta
    def _scale_y(self, y):
        return (2 * (y - self.ymin) / (self.ymax - self.ymin) - 1) * self.out_range

    def _unscale_y(self, z):
        return (z / self.out_range + 1) / 2 * (self.ymax - self.ymin) + self.ymin

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        rng = np.random.default_rng(self.seed)
        self.imp = SimpleImputer(strategy="median", add_indicator=True)
        self.qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=self.seed)
        Xt = self.qt.fit_transform(self.imp.fit_transform(X)).astype("float32")
        y = np.asarray(y, dtype="float64")
        self.ymin, self.ymax = float(y.min()), float(y.max())
        z = self._scale_y(y).astype("float32")

        idx = rng.permutation(len(Xt))
        nv = int(len(Xt) * self.val_frac)
        va, tr = idx[:nv], idx[nv:]
        Xtr, ztr = torch.from_numpy(Xt[tr]), torch.from_numpy(z[tr])
        Xva, zva = torch.from_numpy(Xt[va]), torch.from_numpy(z[va])

        self.model = TanhMLP(Xt.shape[1], self.hidden, self.dropout)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(Xtr))
            for s in range(0, len(perm), self.batch):
                b = perm[s:s + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.model(Xtr[b]), ztr[b])
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                val = loss_fn(self.model(Xva), zva).item()
            if val < best - 1e-6:
                best, bad = val, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        Xt = torch.from_numpy(self.qt.transform(self.imp.transform(X)).astype("float32"))
        self.model.eval()
        with torch.no_grad():
            z = self.model(Xt).numpy().astype("float64")
        return self._unscale_y(z)
