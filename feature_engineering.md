# 特征工程说明

## 一、原始特征保留

以下原始列直接保留，未做额外处理：

| 特征             | 说明                       |
|------------------|----------------------------|
| `name`           | 汽车交易名称编码           |
| `model`          | 车型编码                   |
| `brand`          | 汽车品牌编码               |
| `bodyType`       | 车身类型                   |
| `fuelType`       | 燃油类型                   |
| `gearbox`        | 变速箱类型                 |
| `kilometer`      | 汽车行驶里程（万公里）     |
| `regionCode`     | 地区编码                   |
| `seller`         | 销售方                     |
| `offerType`      | 报价类型                   |
| `v_0` ~ `v_14`   | 匿名脱敏特征（共 15 个）   |

## 二、数据清洗

| 字段                 | 处理方式                                                       |
|----------------------|----------------------------------------------------------------|
| `notRepairedDamage`  | 将 `-` 替换为 NaN，再转为数值型供树模型使用                    |
| `fuelType`           | 用众数填充缺失值                                               |
| `bodyType`           | 用众数填充缺失值                                               |
| `gearbox`            | 用众数填充缺失值                                               |
| `power`              | 将异常值 0 替换为 NaN，截断到 600，再用中位数填充              |
| `model`              | 将 0.0 视为缺失（设为 NaN，不填充）                            |

## 三、日期特征衍生

原始日期列 `regDate` 和 `creatDate` 格式为 YYYYMMDD（如 20040402），拆分后删除原始列。

| 衍生特征       | 计算方式                             | 说明           |
|----------------|--------------------------------------|----------------|
| `reg_year`     | `regDate[:4]`                        | 注册年份       |
| `reg_month`    | `regDate[4:6]`                       | 注册月份       |
| `creat_year`   | `creatDate[:4]`                      | 上架年份       |
| `creat_month`  | `creatDate[4:6]`                     | 上架月份       |
| `car_age_days` | `creatDate - regDate`（精确天数差）  | 车龄（天数）   |
| `car_age_year` | `car_age_days // 365`                | 车龄（年）     |
| `km_per_year`  | `kilometer / (car_age_days / 365)`   | 每年行驶公里数 |

## 四、分桶特征

将连续特征离散化为有序分桶，作为 CatBoost 分类特征（`cat_features`）输入。

### 4.1 power 分桶（`power_bucket`）

| 分桶值              | power 范围   | 含义            |
|---------------------|-------------|-----------------|
| `01_Micro_Electric` | ≤ 60        | 微型/电动车     |
| `02_Economy`        | 61 ~ 100    | 经济型          |
| `03_Best_Seller`    | 101 ~ 180   | 主流畅销型      |
| `04_Premium`        | 181 ~ 300   | 中高端          |
| `05_Performance`    | 301 ~ 500   | 性能车          |
| `06_Hypercar_Exotic`| > 500       | 超跑/ exotic    |

### 4.2 kilometer 分桶（`kilometer_bucket`）

单位为万公里。

| 分桶值                | kilometer 范围 | 含义               |
|-----------------------|---------------|--------------------|
| `01_Showroom`         | < 0.1         | 准新车/展厅车      |
| `02_Nearly_New`       | 0.1 ~ 1.0     | 近新车             |
| `03_Prime`            | 1.0 ~ 3.0     | 黄金车况           |
| `04_Normal`           | 3.0 ~ 6.0     | 正常里程           |
| `05_Old`              | 6.0 ~ 10.0    | 较旧               |
| `06_High_Mileage`     | 10.0 ~ 20.0   | 高里程             |
| `07_Scrap_or_RideHailing` | ≥ 20.0   | 报废/网约车级别    |

## 五、多项式交叉特征

经 `screen_poly_features.py` 筛选出的 Top 2 交叉特征：

| 衍生特征        | 计算方式      | 说明              |
|-----------------|--------------|-------------------|
| `v_0_plus_v_12` | `v_0 + v_12` | 匿名特征加法组合  |
| `v_5_x_v_12`   | `v_5 * v_12` | 匿名特征乘法组合  |

## 五-B、行统计与交互特征（v5 新增）

| 衍生特征          | 计算方式                                | 与 price 相关性 | 说明                         |
|-------------------|----------------------------------------|----------------|------------------------------|
| `v_row_std`       | `v_0 ~ v_14 的行标准差`                | 0.72           | 配置丰富度/离散度            |
| `power_age_ratio` | `power / car_age_year`                 | 0.50           | 性能老化比                   |
| `dmg_x_age`       | `notRepairedDamage × car_age_year`     | -0.20          | 损坏对新车价格打击更大       |
| `brand_dmg_rate`  | 按 brand 聚合的 notRepairedDamage 均值 | —              | 品牌损坏率（3.7%~24%）       |

## 六、目标编码（Target Encoding）

所有聚合统计仅从训练集计算，通过键列映射到训练集和测试集（避免数据泄露）。

### 6.1 brand 级别

基于 `brand` 与 `price` 计算统计特征（v5 去掉 min/max/count，只保留 mean/median/std）。

| 衍生特征             | 计算方式                   |
|----------------------|----------------------------|
| `brand_price_mean`   | 按品牌分组的 price 均值    |
| `brand_price_median` | 按品牌分组的 price 中位数  |
| `brand_price_std`    | 按品牌分组的 price 标准差  |

### 6.2 model 级别（v5 新增）

基于 `model` 与 `price` 计算统计特征。与 brand×model 互补，为罕见 brand-model 组合提供更稳定的基线。

| 衍生特征              | 计算方式                   |
|-----------------------|----------------------------|
| `model_price_mean`    | 按车型分组的 price 均值    |
| `model_price_median`  | 按车型分组的 price 中位数  |
| `model_price_std`     | 按车型分组的 price 标准差  |

### 6.3 bodyType 级别（v5 新增）

基于 `bodyType` 与 `price` 计算统计特征。捕获车身类型的市场均价差异。

| 衍生特征              | 计算方式                     |
|-----------------------|------------------------------|
| `body_price_mean`     | 按车身类型分组的 price 均值  |
| `body_price_median`   | 按车身类型分组的 price 中位数 |
| `body_price_std`      | 按车身类型分组的 price 标准差 |

### 6.4 brand × gearbox（v5 新增）

基于 `(brand, gearbox)` 与 `price` 计算统计特征。捕获品牌 × 变速箱的溢价差异。

| 衍生特征                  | 计算方式                              |
|---------------------------|---------------------------------------|
| `brand_gear_price_mean`   | 按 (brand, gearbox) 分组的 price 均值 |
| `brand_gear_price_median` | 按 (brand, gearbox) 分组的 price 中位数 |
| `brand_gear_price_std`    | 按 (brand, gearbox) 分组的 price 标准差 |

### 6.5 brand x model 交叉聚合

按 `(brand, model)` 对 `price`、`power`、`kilometer` 做 `mean`、`median`、`std` 聚合，共 9 个特征。

| 衍生特征示例                 | 计算方式                                    |
|------------------------------|---------------------------------------------|
| `price_mean`                 | 按 (brand, model) 分组的 price 均值         |
| `power_std`                  | 按 (brand, model) 分组的 power 标准差       |
| `kilometer_median`           | 按 (brand, model) 分组的 kilometer 中位数   |
| ...其余 6 个                  | 同理                                        |

### 6.6 brand x car_age_year 交叉聚合

按 `(brand, car_age_year)` 对 `price`、`power`、`kilometer` 做 `mean`、`median`、`std` 聚合，共 9 个特征。列名前缀为 `brand_age_`。

| 衍生特征示例                     | 计算方式                                         |
|----------------------------------|--------------------------------------------------|
| `brand_age_price_mean`           | 按 (brand, car_age_year) 分组的 price 均值       |
| `brand_age_power_std`            | 按 (brand, car_age_year) 分组的 power 标准差     |
| `brand_age_kilometer_median`     | 按 (brand, car_age_year) 分组的 kilometer 中位数 |
| ...其余 6 个                      | 同理                                             |

## 七、目标变量变换

`price` 呈严重右偏分布（偏度 3.35），使用 `log1p` 变换缓解长尾问题（仅 v3 版本）。

| 版本                 | 目标变量            | 反变换            | 损失函数 |
|----------------------|---------------------|-------------------|----------|
| train_catboost2.py   | `price` (原始)      | 无                | MAE      |
| train_catboost3.py   | `log1p(price)`      | `expm1(pred)`     | RMSE     |
| train_catboost4.py   | `price` (原始)      | 无                | MAE      |
| train_catboost5.py   | `price` (原始)      | 无                | MAE      |
| train_ensemble.py    | `price` (原始)      | 无                | MAE      |

## 八、删除的列

| 列          | 原因                                           |
|-------------|------------------------------------------------|
| `SaleID`    | 交易ID，无预测价值                             |
| `price`     | 目标变量，不作为特征                           |
| `regDate`   | 已衍生为 reg_year/month 等列，避免重复         |
| `creatDate` | 已衍生为 creat_year/month 等列，避免重复       |

## 九、建模流程（单模型 v4/v5）

1. **特征选择**：除 `SaleID` 和 `price` 外的全部列作为特征（v4 共 62 个，v5 共 72 个）；`power_bucket` 和 `kilometer_bucket` 声明为 CatBoost 分类特征（`cat_features`）
2. **数据划分**：15% 验证集 + 85% 训练子集（`random_state=42`）
3. **第一阶段**：用 early_stopping（`od_wait=100`）确定最优迭代数 `best_iteration`
4. **第二阶段**：用 `best_iteration` 在全量训练集上重新训练
5. **预测输出**：`cb_submit_predictions.csv`（UTF-8，表头 SaleID/price，价格下限 max(pred, 1.0)）

## 十、多模型融合 v3（train_ensemble.py）

### 10.1 模型配置

三个模型超参数对齐（depth=7, lr=0.03, l2/lambda=3.0），仅框架和目标变换不同：

| 模型 | 分类特征处理 | 目标变量 | early_stopping |
|------|-------------|---------|----------------|
| CatBoost | 原生 `cat_features` | `price`（原始） | od_wait=100 |
| LightGBM | label encoding + `categorical_feature` | `log1p(price)` | 100 rounds |
| XGBoost | label encoding | `price`（原始） | 100 rounds |

LightGBM 使用 `log1p` 目标变换增加模型多样性，预测时 `expm1` 还原。

### 10.2 v3 改进（当前版本）

相比 v1 的核心改动：

| 改动 | 说明 |
|------|------|
| **CV Target Encoding** | 聚合特征在每折的训练子集上计算，无数据泄露 |
| **StratifiedKFold** | 按 price 10 分位数分层抽样，折间价格分布一致 |
| **每折不同种子** | 5 折各用 `[42, 123, 2024, 666, 999]`，种子多样性 + 折内编码一次完成 |
| **特征筛选** | 过滤 11 个低重要性特征（seller/offerType 等），63 → 65 个特征 |
| **新增交叉特征** | `v_3×v_8`（corr=-0.83）、`v_3×v_12`（corr=-0.70） |
| **新增分桶** | `age_bucket`（6 桶） |
| **LightGBM 正则** | 增加 `reg_alpha=0.5`、`min_child_samples=30` |
| **XGBoost 正则** | 增加 `reg_alpha=0.5`、`min_child_weight=10`、`gamma=0.1` |

### 10.3 训练流程

1. 5 折 StratifiedKFold（按 price 分 10 箱分层）
2. 每折：训练子集算聚合特征（CV Target Encoding）→ 训练 CB/LGB/XGB
3. 测试集预测取 5 折平均
4. 网格搜索 + Nelder-Mead 求最优融合权重
5. 输出加权平均和 Stacking（Ridge）两种结果

### 10.4 线上得分对比

| 版本 | 方案 | 线上 MAE |
|------|------|---------|
| v4 | CatBoost 单模型, lr=0.015, 62 特征 | 454.58 |
| v5 | CatBoost 单模型, lr=0.03, 72 特征 | 450.04 |
| ensemble v1 | 三模型融合, 5×5 折 CV, 72 特征 | 445.93 |
| ensemble v3 (stacking) | CV Target Encoding + StratifiedKFold + log1p, 65 特征 | 444.66 |
| **ensemble v3 (weighted)** | **同上，加权平均融合** | **439.01** |

## 十一、MLP 独立调优（最终方案）

树模型到顶后，转向纯 MLP 方案。去掉 CB/LGB/XGB，只训练多个不同种子的 MLP 取平均。

### 11.1 MLP 调参历程

| 配置 | 每折 MLP 数 | OOF MAE | 线上 MAE |
|------|------------|---------|---------|
| [512, 256] dropout=0.3 lr=1e-3 epochs=300 | 1 | 456.52 | — |
| [512, 256, 128] dropout=0.3 lr=1e-3 epochs=300 | 1 | 454.72 | — |
| [512, 256, 128] dropout=0.2 lr=1e-3 epochs=500 | 1 | 449.54 | — |
| [768, 384, 128] dropout=0.2 lr=5e-4 epochs=1000 | 3 | 432.93 | 421.30 |
| **[1024, 512, 256] dropout=0.2 lr=5e-4 epochs=1000** | **5** | **427.03** | **418.78** |

### 11.2 最终 MLP 配置

| 配置项 | 值 |
|--------|-----|
| 网络结构 | 3 层全连接：input → 1024 → 512 → 256 → 1 |
| 激活函数 | ReLU |
| 正则化 | BatchNorm1d + Dropout(0.2) + weight_decay=1e-4 |
| 损失函数 | L1Loss（MAE） |
| 优化器 | Adam, lr=5e-4 |
| 学习率调度 | CosineAnnealing |
| 早停 | patience=100 |
| 多种子平均 | 每折 5 个不同 seed 的 MLP 取平均 |
| 特征预处理 | StandardScaler 标准化 + NaN 填 0 |
| 硬件 | GPU（CUDA 12.8, RTX 5070 Ti） |

### 11.3 关键发现

- **网络宽度比深度重要**：[256, 128] 欠拟合，[1024, 512, 256] 最优
- **多种子平均收益巨大**：单 MLP OOF 456 → 5 取平均 OOF 427，降 29 分
- **CB/LGB 融合收益递减**：MLP OOF 领先树模型 37+ 分，融合权重太小，不值得 3+ 小时训练
- **log1p 变换对 MLP 无效**：log 空间 MSE 与原始空间 MAE 目标不一致，性能反而下降

### 11.4 失败尝试

| 尝试 | 结果 | 原因 |
|------|------|------|
| name 编码聚合 | 所有模型 MAE 崩到 3000+ | name 唯一值过多，groupby 产生大量噪声组 |
| regionCode 目标编码 | 线上分数下降 | 增加的特征引入噪声 |
| log1p 目标变换 + MSELoss | OOF 468+，远差于原始 | log 空间优化目标与原始空间 MAE 不一致 |
| 缩小网络 [256, 128] | OOF 480 | 欠拟合，表达能力不足 |

## 十二、线上得分对比

| 版本 | 方案 | 线上 MAE |
|------|------|---------|
| v4 | CatBoost 单模型, lr=0.015, 62 特征 | 454.58 |
| v5 | CatBoost 单模型, lr=0.03, 72 特征 | 450.04 |
| ensemble v1 | 三模型融合, 5×5 折 CV, 72 特征 | 445.93 |
| ensemble v3 (stacking) | CV Target Encoding + StratifiedKFold + log1p, 65 特征 | 444.66 |
| ensemble v3 (weighted) | 同上，加权平均融合 | 439.01 |
| ensemble v4 (weighted) | 四模型融合（+MLP [512,256]），65 特征 | 430.13 |
| **MLP 最终版** | **[1024, 512, 256] × 5 种子平均，65 特征** | **418.78** |
