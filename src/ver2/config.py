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
SEQ_LEN = 365                   # 编码器序列长度（天）
QUANTILES = [0.1, 0.5, 0.9]

# 时间切分
TRAIN_END = "2019-12-31"
VAL_END   = "2022-12-31"

# 输出目录
OUT_DIR = BASE_DIR / "outputs_ver2"

# 训练参数
EPOCHS = 60
BATCH_SIZE = 32
HIDDEN_SIZE = 256
NUM_LAYERS = 2
LR = 5e-4
DROPOUT = 0.3
NHEAD = 8
GRAD_CLIP = 5.0

# 损失加权与校准
# q50 的 MSE 辅助损失权重
MSE_AUX_WEIGHT = 0.2
# 步长加权模式：'none' | 'linear' | 'square'
STEP_WEIGHT_MODE = 'linear'
# 线性权重基数：w_h = 1 + STEP_WEIGHT_ALPHA * h
STEP_WEIGHT_ALPHA = 0.1
# 是否在测试集应用基于验证集的 q50 线性校准
CALIBRATE_Q50 = True
