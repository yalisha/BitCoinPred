#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# 数据源（沿用 ver1 的合并数据）
DATA_CSV = BASE_DIR / "data" / "macro_btc_2014_2025_daily.csv"

# // 预测设置
# // 继续预测价格，但使用“相对对数价格”作为目标：y_h = log(C_{t+h}) - log(C_t)
# // 评估与可视化时将映射回价格：Ĉ_{t+h} = C_t * exp(ŷ_h)
PRED_TARGET = "rel_logprice"  # price | logprice | logret | pctret | rel_logprice
HORIZON = 5                    # 多步预测长度（天）
SEQ_LEN = 60                   # 编码器序列长度（天）
QUANTILES = [0.1, 0.5, 0.9]

# 时间切分
TRAIN_END = "2019-12-31"
VAL_END   = "2022-12-31"

# 输出目录
OUT_DIR = BASE_DIR / "outputs_ver2"

# 训练参数
EPOCHS = 80
BATCH_SIZE = 24
HIDDEN_SIZE = 128
LR = 3e-4
DROPOUT = 0.2
NHEAD = 8
LSTM_LAYERS = 2
ATTN_DROPOUT = 0.2
FF_DROPOUT = 0.15
GRAD_CLIP = 3.0
WEIGHT_DECAY = 1e-5

# 时间序列交叉验证
CV_ENABLED = True
CV_FOLDS = 4
CV_EPOCHS = 40
CV_MIN_TRAIN_SAMPLES = 200
CV_VAL_SAMPLES = 0  # 0 表示使用验证集样本数作为窗口

# 超参搜索（Optuna）
ENABLE_OPTUNA = True
OPTUNA_TRIALS = 20
OPTUNA_TIMEOUT = 1800  # 秒，0 表示不限
OPTUNA_STUDY = "tft_ver2"
OPTUNA_STORAGE = None  # 例如 "sqlite:///tft_ver2.db"

# 损失加权与校准
# q50 的 MSE 辅助损失权重
MSE_AUX_WEIGHT = 0.2
# 步长加权模式：'none' | 'linear' | 'square'
STEP_WEIGHT_MODE = 'linear'
# 线性权重基数：w_h = 1 + STEP_WEIGHT_ALPHA * h
STEP_WEIGHT_ALPHA = 0.1
# 是否在测试集应用基于验证集的 q50 线性校准
CALIBRATE_Q50 = True
