#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import nn


def _to_t(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32))


class GLU(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, 2*d)
    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    def __init__(self, d_in: int, d_h: int, d_out: int, dropout=0.1):
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.fc1 = nn.Linear(d_in, d_h)
        self.fc2 = nn.Linear(d_h, d_out)
        self.glu = GLU(d_out)
        self.ln  = nn.LayerNorm(d_out)
        self.do  = nn.Dropout(dropout)
    def forward(self, x):
        y = torch.nn.functional.elu(self.fc1(x))
        y = self.fc2(y)
        y = self.do(y)
        y = self.glu(y)
        return self.ln(self.skip(x) + y)


class VSN(nn.Module):
    """变量选择网络：对每个变量独立投影+GRN，再用权重网络做softmax聚合。"""
    def __init__(self, num_vars: int, d_model: int, d_hidden: int, time_distributed: bool = True):
        super().__init__()
        self.enc = nn.ModuleList([nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), GRN(d_model, d_hidden, d_model)) for _ in range(num_vars)])
        self.w_grn = GRN(num_vars, d_hidden, num_vars)
        self.time_distributed = time_distributed

    def forward(self, x):
        # x: [B, T, F]
        B, T, F = x.shape
        hs = []
        for i in range(F):
            h = self.enc[i](x[..., i:i+1])  # [B, T, D]
            hs.append(h)
        H = torch.stack(hs, dim=2)  # [B, T, F, D]
        # 权重
        w_in = x if self.time_distributed else x.mean(dim=1, keepdim=True)
        w = self.w_grn(w_in)  # [B,T,F] or [B,1,F]
        w = torch.softmax(w, dim=-1)
        if not self.time_distributed:
            w = w.repeat(1, T, 1)
        z = torch.sum(H * w.unsqueeze(-1), dim=2)  # [B, T, D]
        return z, w


@dataclass
class TFTCfg:
    d_model: int = 64
    d_hidden: int = 128
    nhead: int = 4
    dropout: float = 0.1
    horizon: int = 14
    quantiles: List[float] = None
    device: str = "cpu"


class TFTMultiHQuantile(nn.Module):
    def __init__(self, num_obs: int, num_kn: int, num_static: int, cfg: TFTCfg):
        super().__init__()
        self.cfg = cfg
        self.vsn_obs = VSN(num_obs, cfg.d_model, cfg.d_hidden, time_distributed=True)
        self.vsn_kn  = VSN(num_kn,  cfg.d_model, cfg.d_hidden, time_distributed=True)
        self.vsn_st  = VSN(num_static, cfg.d_model, cfg.d_hidden, time_distributed=False)

        self.enc_lstm = nn.LSTM(cfg.d_model, cfg.d_model, batch_first=True)
        self.dec_lstm = nn.LSTM(cfg.d_model, cfg.d_model, batch_first=True)

        self.attn_norm = nn.LayerNorm(cfg.d_model)
        self.cross_attn = nn.MultiheadAttention(cfg.d_model, cfg.nhead, batch_first=True, dropout=cfg.dropout)
        self.attn_glu  = GLU(cfg.d_model)
        self.out_norm  = nn.LayerNorm(cfg.d_model)

        Q = len(cfg.quantiles)
        H = cfg.horizon
        self.head = nn.Linear(cfg.d_model, Q)

    def forward(self, X_obs, X_kn, X_static):
        # X_obs: [B, T_enc, F_obs], X_kn: [B, T_dec, F_kn], X_static: [B, F_st]
        z_obs, w_obs = self.vsn_obs(X_obs)
        z_kn,  w_kn  = self.vsn_kn(X_kn)
        z_st,  w_st  = self.vsn_st(X_static.unsqueeze(1))  # [B,1,D]

        enc_out, (h, c) = self.enc_lstm(z_obs)
        # 将静态上下文注入decoder初始状态（简化：线性投影叠加）
        dec_in = z_kn
        dec_out, _ = self.dec_lstm(dec_in, (h, c))

        # Cross-attention: queries=dec_out, keys/values=enc_out
        q = self.attn_norm(dec_out)
        attn_out, attn_w = self.cross_attn(q, enc_out, enc_out)
        attn_out = self.attn_glu(attn_out)
        h = self.out_norm(dec_out + attn_out)
        q_pred = self.head(h)  # [B, H, Q]
        # 防止分位交叉：按最后维排序
        q_pred_sorted, _ = torch.sort(q_pred, dim=-1)
        return q_pred_sorted, {"w_obs": w_obs, "w_kn": w_kn, "w_st": w_st, "attn": attn_w}


def pinball_loss(y_true: torch.Tensor, y_pred_q: torch.Tensor, quantiles: List[float], step_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """分位损失，支持按步长加权。
    y_true: [B,H]
    y_pred_q: [B,H,Q]
    step_weights: [H] or None
    """
    losses = []
    for i, q in enumerate(quantiles):
        e = y_true - y_pred_q[..., i]  # [B,H]
        pin = torch.maximum(q*e, (q-1)*e)  # [B,H]
        if step_weights is not None:
            pin = pin * step_weights.view(1, -1)
        losses.append(pin.mean())
    return torch.stack(losses).mean()


def train_one(cfg: TFTCfg,
              model: TFTMultiHQuantile,
              tr: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
              va: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
              epochs: int = 20, lr: float = 1e-3, batch_size: int = 128,
              device: str = "cpu",
              step_weights_np: Optional[np.ndarray] = None,
              mse_aux_weight: float = 0.1):
    Xo_tr, Xk_tr, Xs_tr, y_tr = tr
    Xo_va, Xk_va, Xs_va, y_va = va
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Some torch versions don't support 'verbose' arg; omit for compatibility
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
    best = float("inf")
    best_state = None
    patience = 5
    wait = 0

    def batched(Xo, Xk, Xs, y, shuffle=True):
        idx = np.arange(len(y))
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(y), batch_size):
            sel = idx[i:i+batch_size]
            yield (_to_t(Xo[sel]).to(device), _to_t(Xk[sel]).to(device), _to_t(Xs[sel]).to(device), _to_t(y[sel]).to(device))

    q_index = (cfg.quantiles.index(0.5) if (cfg.quantiles and 0.5 in cfg.quantiles) else (len(cfg.quantiles)//2 if cfg.quantiles else 0))
    mse_aux_weight = float(mse_aux_weight)
    step_weights_t = None
    if step_weights_np is not None:
        step_weights_t = torch.from_numpy(step_weights_np.astype(np.float32)).to(device)

    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for Xo, Xk, Xs, y in batched(Xo_tr, Xk_tr, Xs_tr, y_tr, shuffle=True):
            opt.zero_grad()
            q_pred, _ = model(Xo, Xk, Xs)
            loss_pin = pinball_loss(y, q_pred, cfg.quantiles, step_weights=step_weights_t)
            # q50 的轻量 MSE 辅助，稳定中位输出
            if q_pred.shape[-1] > q_index:
                q50 = q_pred[..., q_index]
                loss_mse = torch.mean((q50 - y)**2)
                loss = loss_pin + mse_aux_weight * loss_mse
            else:
                loss = loss_pin
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(y)
            n_tr += len(y)
        tr_loss /= max(1, n_tr)

        # val
        model.eval()
        with torch.no_grad():
            Xo = _to_t(Xo_va).to(device)
            Xk = _to_t(Xk_va).to(device)
            Xs = _to_t(Xs_va).to(device)
            y  = _to_t(y_va).to(device)
            q_pred, _ = model(Xo, Xk, Xs)
            loss_pin = pinball_loss(y, q_pred, cfg.quantiles, step_weights=step_weights_t)
            if q_pred.shape[-1] > q_index:
                q50 = q_pred[..., q_index]
                loss_mse = torch.mean((q50 - y)**2)
                va_loss = (loss_pin + mse_aux_weight * loss_mse).item()
            else:
                va_loss = loss_pin.item()

        if va_loss < best - 1e-6:
            best = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
        scheduler.step(va_loss)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model: TFTMultiHQuantile,
            Xo: np.ndarray, Xk: np.ndarray, Xs: np.ndarray,
            device: str = "cpu") -> Tuple[np.ndarray, dict]:
    model.eval()
    with torch.no_grad():
        Xo_t = _to_t(Xo).to(device)
        Xk_t = _to_t(Xk).to(device)
        Xs_t = _to_t(Xs).to(device)
        q_pred, details = model(Xo_t, Xk_t, Xs_t)
    return q_pred.cpu().numpy(), {k: (v.cpu().numpy() if hasattr(v, 'cpu') else v) for k, v in details.items()}
