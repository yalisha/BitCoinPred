# Gold CNN-BiLSTM 模块拆解

## 目标概述
- 复现 Amini & Kalantari (2024) 的 CNN-BiLSTM 黄金价格预测方法。
- 使用 44 年黄金日度数据构建教学友好的代码结构。
- 支持本地 CSV 与在线数据源，提供自动调参与评估工具。

## 数据模块 `goldpred.data`
- `load_gold_history`：统一本地 CSV、Yahoo Finance、FRED 等来源，标准化列名和索引。
- 自动按照配置筛选日期区间、补齐缺失值，输出带 `close/open/high/low` 的 DataFrame。
- 结合 `config.DataSourceConfig` 可快速切换数据粒度、日期范围。

## 特征模块 `goldpred.features`
- `build_feature_frame`：基于 `FeatureConfig` 生成收益率、波动率、移动平均、RSI 等技术指标。
- 模块化开关允许课堂演示不同特征组合对模型的影响。
- 清洗后的特征 DataFrame 直接供序列数据集使用。

## 数据集与切分 `goldpred.training.dataset`
- `SequenceDataset`：将特征矩阵转换为定长滑窗与预测跨度 (lookback/horizon)。
- 内部完成张量化，便于与 PyTorch 训练循环衔接。

## 训练管线 `goldpred.training.loop`
- `Trainer`：封装训练/验证循环、MSE 损失、早停与梯度裁剪。
- `create_dataloaders`：使用时间顺序切分训练/验证/测试集，避免信息泄露。
- `TrainingRecord`：保存每个 epoch 的损失与 RMSE/MAE/MAPE 指标。

## 模型模块 `goldpred.models`
- `CNNBiLSTMModel`：包含多层一维卷积特征提取、双向 LSTM 编码与全连接回归头。
- 通过 `ModelConfig` 控制卷积核大小、通道数、LSTM 层数等超参数。

## 自动调参 `goldpred.tuning`
- `run_hyperparameter_search`：Optuna 搜索 lookback、dropout、学习率等关键超参数。
- 使用复制配置对象方式避免污染默认配置，兼容 CPU 运行。

## 端到端管线 `goldpred.pipeline`
- `run_pipeline`：连通数据、特征、模型、训练与评估，返回训练历史、测试指标和预测序列。
- `PipelineResult`：整理结果，便于教程中展示损失曲线、预测对比。

## 脚本工具 `goldpred.scripts`
- `prepare_data.py`：快速生成特征数据并落盘（CSV/Parquet），方便学生预先下载或复现。

## 教学建议
1. 课堂先运行 `prepare_data.py` 演示特征文件生成；
2. 使用 `run_pipeline` Notebook 展示模型训练与指标；
3. 最后引导学员尝试 `run_hyperparameter_search` 观察超参对结果的影响。
