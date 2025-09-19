#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""生成 44 年黄金价格的特征数据集，供后续训练使用。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import CONFIG
from ..data import load_gold_history
from ..features import build_feature_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=CONFIG.data.csv_path.with_name("gold_features"),
        help="输出特征数据路径 (无需扩展名)",
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="输出文件格式",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    price_df = load_gold_history(CONFIG.data)
    features = build_feature_frame(price_df, config=CONFIG.features)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "csv":
        output_path = output_path.with_suffix(".csv")
        features.to_csv(output_path)
    else:
        output_path = output_path.with_suffix(".parquet")
        features.to_parquet(output_path)
    print(f"[prepare_data] 特征数据写入 {output_path}，共 {len(features):,} 条记录。")


if __name__ == "__main__":  # pragma: no cover
    main()
