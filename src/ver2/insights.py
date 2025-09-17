#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def _topk(labels: list[str], values: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
    order = np.argsort(-values)
    return [(labels[i], float(values[i])) for i in order[:k]]


def save_variable_weights(base_path: Path,
                          split: str,
                          score: dict,
                          title_prefix: str = ""):  # score: {feature_names, time_mean, mean, top_features}
    if not score:
        return
    feature_names = score.get("feature_names") or []
    feature_names = list(feature_names)
    mean = np.asarray(score.get("mean"), dtype=float)
    time_mean = np.asarray(score.get("time_mean"), dtype=float)
    split_dir = base_path / "insights" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "var_weights_mean.npy", mean)
    np.save(split_dir / "var_weights_time.npy", time_mean)
    with (split_dir / "var_weights_topk.txt").open("w", encoding="utf-8") as f:
        for name, value in _topk(feature_names, mean):
            f.write(f"{name}\t{value:.6f}\n")


def save_attn_weights(base_path: Path,
                      split: str,
                      attn_summary: dict):
    if not attn_summary:
        return
    split_dir = base_path / "insights" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    mean = np.asarray(attn_summary.get("mean"), dtype=float)
    np.save(split_dir / "attention_mean.npy", mean)


def plot_variable_bar(feature_names: Iterable[str], values: np.ndarray, title: str, out_path: Path, top_k: int = 20):
    import matplotlib.pyplot as plt

    idx = np.argsort(-values)[:top_k]
    labels = [feature_names[i] for i in idx]
    vals = values[idx]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(idx)), vals)
    plt.xticks(range(len(idx)), labels, rotation=60, ha='right', fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_time_heatmap(values: np.ndarray, title: str, out_path: Path, xlabel: str = "Time", ylabel: str = "Feature"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.imshow(values.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_attention_heatmap(values: np.ndarray, title: str, out_path: Path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    plt.imshow(values, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

