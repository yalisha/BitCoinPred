#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# 根目录：.../2比特币预测
BASE_DIR = Path(__file__).resolve().parents[1]

# 数据文件（由 data/Other.py 生成）
DEFAULT_DATA = BASE_DIR / "data" / "macro_btc_2014_2025_daily.csv"

# 预测步长（天）：不使用任何虚拟变量
HORIZON = 1

# 时间切分（可按论文设定调整）
TRAIN_END = "2019-12-31"
VAL_END = "2022-12-31"  # 验证集结束；测试集为其后所有可用

# 输出目录
OUT_DIR = BASE_DIR / "outputs"

