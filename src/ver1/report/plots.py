#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 兼容无图形界面环境
import matplotlib.pyplot as plt


def _ensure_figsize(figsize=None):
    return figsize if figsize is not None else (10, 4)


def plot_timeseries(dates: Iterable, y_true: Iterable, y_pred: Iterable, title: str, outfile: Path):
    dates = pd.to_datetime(pd.Series(dates))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    plt.figure(figsize=_ensure_figsize())
    plt.plot(dates, y_true, label='True', lw=1.5)
    plt.plot(dates, y_pred, label='Pred', lw=1.2)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Target')
    plt.legend()
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_scatter(y_true: Iterable, y_pred: Iterable, title: str, outfile: Path):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    lim = [np.min([y_true.min(), y_pred.min()]), np.max([y_true.max(), y_pred.max()])]
    plt.figure(figsize=_ensure_figsize((5, 5)))
    plt.scatter(y_true, y_pred, s=8, alpha=0.6)
    plt.plot(lim, lim, 'r--', lw=1)
    plt.title(title)
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_residuals(dates: Iterable, residuals: Iterable, title: str, outfile: Path):
    dates = pd.to_datetime(pd.Series(dates))
    residuals = np.asarray(residuals)
    plt.figure(figsize=_ensure_figsize())
    plt.plot(dates, residuals, lw=1.0)
    plt.axhline(0.0, color='k', lw=0.8)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_hist(data: Iterable, title: str, outfile: Path):
    data = np.asarray(data)
    plt.figure(figsize=_ensure_figsize())
    plt.hist(data, bins=40, alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_feature_importance(names: List[str], scores: Iterable, title: str, outfile: Path, topk: int = 20):
    names = np.asarray(names)
    scores = np.asarray(scores)
    order = np.argsort(np.abs(scores))[::-1][:topk]
    names_top = names[order]
    scores_top = scores[order]

    plt.figure(figsize=_ensure_figsize((8, 6)))
    y = np.arange(len(names_top))
    plt.barh(y, scores_top)
    plt.yticks(y, names_top)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_attention_heatmap(attn: np.ndarray, title: str, outfile: Path, vmax: Optional[float] = None):
    """attn: [T, S] 或 [S, S] 的注意力矩阵，绘制热力图。"""
    A = np.asarray(attn)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(A, aspect='auto', origin='lower', cmap='viridis', vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel('Source time')
    plt.ylabel('Target time')
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_var_weights_heatmap(w_time: np.ndarray, feature_names: List[str], title: str, outfile: Path):
    """w_time: [T, F] 每个时间步的变量选择权重（已做平均）；绘制 T×F 热力图。"""
    W = np.asarray(w_time)
    plt.figure(figsize=(max(8, len(feature_names) * 0.4), 5))
    im = plt.imshow(W, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel('Feature')
    plt.ylabel('Time step')
    plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names, rotation=90)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_var_weights_bar(feature_names: List[str], w_mean: np.ndarray, title: str, outfile: Path, topk: int = 20):
    """w_mean: [F] 变量选择权重在(时间×样本)上的平均；绘制Top-K条形图。"""
    names = np.asarray(feature_names)
    scores = np.asarray(w_mean)
    order = np.argsort(scores)[::-1][:topk]
    names_top = names[order]
    scores_top = scores[order]
    plt.figure(figsize=(8, max(5, topk * 0.3)))
    y = np.arange(len(names_top))
    plt.barh(y, scores_top)
    plt.yticks(y, names_top)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()
