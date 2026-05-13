---
title: 二手车价格预测项目
date: 2026-05-11
tags:
  - project
  - machine-learning
  - kaggle
  - catboost
  - lightgbm
  - xgboost
  - mlp
category: project
status: completed
---

# 二手车价格预测项目

> 天池学习赛 | 最终成绩：MAE 418.78（MLP 5 种子平均）
> 代码目录：`D:\CodeProjects\vehicle_price_prediction`

---

## 项目概览

预测二手车的交易价格（回归问题），评价指标为 MAE（平均绝对误差）。

### 数据
- 训练集 15 万条，测试集 5 万条
- 基础字段：brand、model、power、kilometer、regDate、creatDate、bodyType、gearbox、fuelType、notRepairedDamage 等
- 15 个匿名脱敏特征 `v_0` ~ `v_14`
- 目标变量 `price` 严重右偏（偏度 3.35）

---

## 关键经验

### 特征工程

#### 1. 日期特征衍生
- `regDate` / `creatDate`（YYYYMMDD）→ 拆分为 year、month
- 核心衍生：`car_age_days`（车龄天数）、`car_age_year`（车龄年）、`km_per_year`（年里程）

#### 2. 分桶（离散化）
- 连续特征分桶后作为 CatBoost 分类特征输入，效果优于直接用连续值
- 有效分桶：`power_bucket`（6 档）、`kilometer_bucket`（7 档）、`age_bucket`（6 档）
- **经验**：分桶边界要符合业务语义（如功率 60/100/180/300/500 对应微型/经济/畅销/中高端/性能/超跑）

#### 3. 目标编码（Target Encoding）
- 按类别字段对 price 做聚合统计（mean/median/std），信息量很大
- **关键**：必须 CV 内计算（用训练折的聚合映射到验证折和测试集），否则会数据泄露
- 有效维度：brand、model、bodyType、brand×gearbox、brand×model、brand×car_age_year

#### 4. 交叉特征
- 匿名特征之间的乘法和加法组合
- 有效组合：`v_0+v_12`、`v_5×v_12`、`v_3×v_8`、`v_3×v_12`
- **经验**：用残差相关性筛选比穷举快得多

#### 5. 行统计特征
- `v_row_std`（v_0~v_14 行标准差）与 price 相关性 0.72，非常有效

#### 6. 特征筛选
- 用 CatBoost 特征重要性过滤，删掉 importance < 0.05 的特征
- seller、offerType 重要性为 0，果断删除
- 删除 11 个低重要性特征后效果更好（减少噪声）

### 模型训练

#### 三模型融合
| 模型 | 目标变量 | 特点 |
|------|---------|------|
| CatBoost | 原始 price | 原生支持分类特征 |
| LightGBM | log1p(price) | 用 log 变换增加模型多样性 |
| XGBoost | 原始 price | 常规配置 |

#### 融合策略
- **加权平均** > Stacking（Ridge）
- 加权平均用网格搜索 + Nelder-Mead 优化权重
- Ridge Stacking 反而更差（444.66 vs 439.01），可能正则化压缩了有效系数

#### CV 策略
- 5 折 StratifiedKFold（按 price 10 分位数分层）
- 每折不同种子 `[42, 123, 2024, 666, 999]`，兼顾种子多样性和效率
- 测试集预测取 5 折平均

#### 超参数
- 三模型对齐：depth=7, lr=0.03, l2/lambda=3.0
- CatBoost: random_strength=0.8, bagging_temperature=0.8, rsm=0.8
- LightGBM: reg_alpha=0.5, min_child_samples=30, colsample=0.8
- XGBoost: reg_alpha=0.5, min_child_weight=10, gamma=0.1
- **MLP 最终版**: 3层全连接(1024→512→256), Dropout(0.2), Adam lr=5e-4, epochs=1000, patience=100
- 每折 5 个不同种子 MLP 取平均，大幅降低预测方差
- GPU: RTX 5070 Ti, CUDA 12.8

#### MLP 的关键作用
- MLP 经调优后 OOF MAE 427，远超树模型（CB 464, LGB 468）
- 最终放弃树模型融合，纯 MLP 5 种子平均线上 418.78
- 多种子平均是最大收益来源：单 MLP 456 → 5 取平均 427，降 29 分

---

## 版本迭代轨迹

| 版本 | 方案 | 线上 MAE | 主要改动 |
|------|------|---------|----------|
| v4 | CatBoost 单模型 | 454.58 | 基线 |
| v5 | CatBoost 单模型 | 450.04 | +model/bodyType/brand×gear 聚合 + 行统计特征 |
| ensemble v1 | 三模型融合 | 445.93 | +LGB/XGB + 5×5 折 CV |
| ensemble v3 | 加权平均 | 439.01 | +CV Target Encoding + StratifiedKFold + log1p + 特征筛选 |
| **ensemble v4** | **四模型加权（+MLP）** | **430.13** | **+PyTorch MLP 神经网络** |
| **MLP 最终版** | **[1024,512,256] × 5 种子** | **418.78** | **纯 MLP 多种子平均** |

---

## 调参经验总结

- 网络宽度比深度重要：[256,128] 欠拟合，[1024,512,256] 最优
- 多种子平均收益巨大（~29 分），是 MLP 最大的调参杠杆
- log1p 变换对 MLP 无效：log 空间 MSE 与原始空间 MAE 目标不一致
- 树模型到顶后无法追赶 MLP，融合收益递减

---

## 相关文件

| 文件 | 用途 |
|------|------|
| `train_catboost4.py` | v4 单模型基线 |
| `train_catboost5.py` | v5 增强特征单模型 |
| `train_ensemble.py` | v3 三模型融合 |
| `train_ensemble_v4.py` | v4 四模型融合 |
| `tune_mlp.py` | MLP 调参脚本（最终版） |
| `feature_engineering.md` | 特征工程详细文档 |
| `data_fields.md` | 数据字段说明 |
