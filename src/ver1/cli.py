#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from .train import TrainConfig, main_cli
from . import config as default_cfg


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BTC 预测（无虚拟变量版本）")
    ap.add_argument("--data", type=str, default=str(default_cfg.DEFAULT_DATA), help="数据CSV路径")
    ap.add_argument("--h", type=int, default=default_cfg.HORIZON, help="预测步长(日)")
    ap.add_argument("--train_end", type=str, default=default_cfg.TRAIN_END)
    ap.add_argument("--val_end", type=str, default=default_cfg.VAL_END)
    ap.add_argument("--model", type=str, default="ridge", choices=[
        "ridge", "ols", "svr", "xgb", "lgbm", "arima", "varx", "garch", "lstm", "gru", "cnn", "tft", "all"
    ])
    # 序列/深度模型参数
    ap.add_argument("--seq_len", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge 正则系数")
    ap.add_argument("--target", type=str, default="logret", choices=["logret", "pctret", "price"], help="预测目标类型")
    ap.add_argument("--out", type=str, default=str(default_cfg.OUT_DIR), help="输出目录")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        data_path=Path(args.data),
        horizon=args.h,
        train_end=args.train_end,
        val_end=args.val_end,
        model=args.model,
        ridge_alpha=args.alpha,
        out_dir=Path(args.out),
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch,
        hidden_size=args.hs,
        lr=args.lr,
        target_type=args.target,
    )
    main_cli(cfg)


if __name__ == "__main__":
    main()
