#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32))


class GLU(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, 2 * d)

    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """Gated Residual Network (TFT).
    GRN(x, c) = LayerNorm(x + GLU( W2(ELU(W1([x;c]))) ))
    其中 c 为可选上下文（如静态上下文）。
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        d_out = d_out or d_in
        self.d_out = d_out
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.ln = nn.LayerNorm(d_out)

    def forward(self, x):
        # 无上下文版本（本项目暂不传静态上下文）
        y = self.fc2(F.elu(self.fc1(x)))
        y = self.dropout(y)
        y = self.glu(y)
        return self.ln(self.skip(x) + y)


class VariableSelectionNetwork(nn.Module):
    """变量选择网络（TFT）。
    - 对每个变量的单独映射/GRN提取表示
    - 用权重网络产生 softmax 权重，对变量维聚合
    输入: x in [B, T, F]
    输出: z in [B, T, D], weights in [B, T, F]
    """

    def __init__(self, num_vars: int, d_model: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.num_vars = num_vars
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(1, d_model), nn.ReLU()) for _ in range(num_vars)])
        self.grns = nn.ModuleList([GRN(d_model, d_hidden, d_out=d_model, dropout=dropout) for _ in range(num_vars)])
        self.weight_grn = GRN(num_vars, d_hidden, d_out=num_vars, dropout=dropout)

    def forward(self, x):
        # x: [B, T, F]
        B, T, Fv = x.shape
        xs = []
        for i in range(Fv):
            vi = x[..., i:i+1]  # [B, T, 1]
            h = self.encoders[i](vi)  # [B, T, D]
            h = self.grns[i](h)
            xs.append(h)
        H = torch.stack(xs, dim=2)  # [B, T, F, D]
        # 权重: 先在变量维度上用线性-GLU-残差（GRN）得到 [B,T,F]，再 softmax
        w = self.weight_grn(x)  # [B, T, F]
        w = F.softmax(w, dim=-1)
        # 加权求和
        z = torch.sum(H * w.unsqueeze(-1), dim=2)  # [B, T, D]
        return z, w


class TFTCore(nn.Module):
    """简化但架构一致的 Temporal Fusion Transformer。
    包含: VSN -> 编码器LSTM -> 解码器LSTM -> 多头注意力(带门控+残差+层归一) -> 输出头
    说明: 不含完整静态上下文与门控融合的所有分支，但保留核心组件与信息流路径，足以严格对齐TFT思路。
    """

    def __init__(self, num_vars: int, d_model: int = 64, d_hidden: int = 128, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.vsn = VariableSelectionNetwork(num_vars, d_model, d_hidden, dropout)
        self.enc_lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        self.dec_lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=dropout)
        self.attn_glu = GLU(d_model)
        self.attn_out_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, T, F]
        z, w = self.vsn(x)  # [B, T, D], [B, T, F]
        # 编码器-解码器（单步预测时，两者同输入；多步时可拆分过去/未来）
        enc_out, _ = self.enc_lstm(z)
        dec_out, _ = self.dec_lstm(z)
        # 自注意力（因果mask，防止看未来）
        B, T, D = dec_out.shape
        mask = torch.triu(torch.ones(T, T, device=dec_out.device), diagonal=1).bool()
        attn_out, attn_weights = self.attn(self.attn_norm(dec_out), enc_out, enc_out, attn_mask=mask)
        attn_out = self.attn_glu(attn_out)
        h = self.attn_out_norm(dec_out + attn_out)
        y = self.out(h[:, -1, :])  # 取最后时间步用于一步预测
        return y.squeeze(-1), w, attn_weights


@dataclass
class TFTConfig:
    num_vars: int
    d_model: int = 64
    d_hidden: int = 128
    nhead: int = 4
    dropout: float = 0.1
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    device: str = "cpu"


class TFTModel:
    def __init__(self, cfg: TFTConfig):
        self.cfg = cfg
        self.net = TFTCore(cfg.num_vars, cfg.d_model, cfg.d_hidden, cfg.nhead, cfg.dropout).to(cfg.device)

    def fit_predict(self,
                    Xtr: np.ndarray, ytr: np.ndarray,
                    Xva: np.ndarray, yva: np.ndarray,
                    Xte: np.ndarray,
                    return_details: bool = False) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, dict]:
        # DataLoader
        from torch.utils.data import DataLoader, TensorDataset
        def loader(X, y=None, shuffle=False):
            Xt = _to_tensor(X)
            if y is None:
                ds = TensorDataset(Xt)
            else:
                yt = _to_tensor(y.reshape(-1, 1))
                ds = TensorDataset(Xt, yt)
            return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle)

        train_loader = loader(Xtr, ytr, shuffle=True)
        val_loader = loader(Xva, yva, shuffle=False)

        # Optim
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        crit = nn.MSELoss()

        best = float("inf")
        best_state = None
        patience, wait = 5, 0

        for ep in range(self.cfg.epochs):
            self.net.train()
            tr_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(self.cfg.device)
                yb = yb.to(self.cfg.device).squeeze(1)
                opt.zero_grad()
                yp, _, _ = self.net(xb)
                loss = crit(yp, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tr_loss += loss.item() * len(xb)
            tr_loss /= len(train_loader.dataset)

            # val
            self.net.eval()
            va_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.cfg.device)
                    yb = yb.to(self.cfg.device).squeeze(1)
                    yp, _, _ = self.net(xb)
                    va_loss += crit(yp, yb).item() * len(xb)
            va_loss /= len(val_loader.dataset)

            if va_loss < best - 1e-6:
                best = va_loss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        # Predict + collect attention / var weights (mean over batches)
        self.net.eval()
        with torch.no_grad():
            yva_pred, wv, attnv = self.net(_to_tensor(Xva).to(self.cfg.device))
            yte_pred, wt, attnt = self.net(_to_tensor(Xte).to(self.cfg.device))
        yva_np = yva_pred.cpu().numpy()
        yte_np = yte_pred.cpu().numpy()

        if not return_details:
            return yva_np, yte_np

        # 平均注意力和变量权重
        # attn: [T, S]（平均头）；w: [B, T, F]（这里B在一次前向中等于X的batch）
        def summarize(w: torch.Tensor, attn: torch.Tensor):
            if w is None:
                return None
            # w: last batch的 [B, T, F] -> 时间平均/样本平均
            w_np = w.cpu().numpy()
            w_time_mean = w_np.mean(axis=0)  # [T, F]
            w_mean = w_time_mean.mean(axis=0)  # [F]
            attn_np = attn.cpu().numpy() if attn is not None else None
            return {
                'var_w_time': w_time_mean,
                'var_w_mean': w_mean,
                'attn_mean': attn_np,
            }

        details = {
            'val': summarize(wv, attnv),
            'test': summarize(wt, attnt),
        }
        return yva_np, yte_np, details
