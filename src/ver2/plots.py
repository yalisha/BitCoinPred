#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_price_series(dates: pd.Series, y_true: np.ndarray, y_pred: np.ndarray, title: str, outfile: Path):
    plt.figure(figsize=(10,4))
    plt.plot(pd.to_datetime(dates), y_true, label='True', lw=1.6)
    plt.plot(pd.to_datetime(dates), y_pred, label='Pred (q50)', lw=1.2)
    plt.title(title)
    plt.legend(); plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_step_series(dates: pd.Series, true_step: np.ndarray, pred_step: np.ndarray, h: int, title: str, outfile: Path):
    plot_price_series(dates, true_step, pred_step, f"{title} (h={h})", outfile)


def plot_quantile_band(dates: pd.Series, q_low: np.ndarray, q_med: np.ndarray, q_high: np.ndarray, title: str, outfile: Path):
    d = pd.to_datetime(dates)
    plt.figure(figsize=(10,4))
    plt.plot(d, q_med, label='q50', color='C0')
    plt.fill_between(d, q_low, q_high, color='C0', alpha=0.2, label='q10–q90')
    plt.title(title)
    plt.legend(); plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_step_metric(metric_vals: np.ndarray, title: str, outfile: Path):
    import numpy as np
    h = np.arange(1, len(metric_vals)+1)
    plt.figure(figsize=(7,4))
    plt.plot(h, metric_vals, marker='o')
    plt.xlabel('Horizon step (h)')
    plt.ylabel('Metric')
    plt.title(title)
    plt.grid(alpha=0.3)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_bar(values: np.ndarray, title: str, outfile: Path, ylabel: str = ''):
    h = np.arange(1, len(values)+1)
    plt.figure(figsize=(7,4))
    plt.bar(h, values)
    plt.xlabel('Horizon step (h)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.3, axis='y')
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
