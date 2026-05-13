"""
二手车价格预测 - CatBoost 建模 v4
基于 v2，增加 power 分桶特征
流程：15% 验证集 + early_stopping -> best_iteration -> 全量训练集重训 -> 预测
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor, Pool
import warnings

warnings.filterwarnings('ignore')


# ========== power 分桶函数 ==========
def bucket_power(power):
    if power <= 60:
        return '01_Micro_Electric'
    elif power <= 100:
        return '02_Economy'
    elif power <= 180:
        return '03_Best_Seller'
    elif power <= 300:
        return '04_Premium'
    elif power <= 500:
        return '05_Performance'
    else:
        return '06_Hypercar_Exotic'


def bucket_kilometer(km):
    if km < 0.1:
        return '01_Showroom'
    elif km < 1.0:
        return '02_Nearly_New'
    elif km < 3.0:
        return '03_Prime'
    elif km < 6.0:
        return '04_Normal'
    elif km < 10.0:
        return '05_Old'
    elif km < 20.0:
        return '06_High_Mileage'
    else:
        return '07_Scrap_or_RideHailing'


# ========== 1. 加载数据 ==========
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

df_train = pd.read_csv('used_car_train.csv', sep=' ')
df_test = pd.read_csv('used_car_test.csv', sep=' ')

test_SaleID = df_test['SaleID'].values

print(f"训练集: {df_train.shape}")
print(f"测试集: {df_test.shape}")


# ========== 2. 特征工程 ==========
def preprocess(df):
    df = df.copy()

    # --- 日期特征：regDate / creatDate 格式 YYYYMMDD ---
    df['regDate_str'] = df['regDate'].astype(str)
    df['reg_year'] = df['regDate_str'].str[:4].astype(int)
    df['reg_month'] = df['regDate_str'].str[4:6].astype(int)

    df['creatDate_str'] = df['creatDate'].astype(str)
    df['creat_year'] = df['creatDate_str'].str[:4].astype(int)
    df['creat_month'] = df['creatDate_str'].str[4:6].astype(int)

    # 车龄（天数）= creatDate - regDate
    reg_dt = pd.to_datetime(df['regDate_str'], format='%Y%m%d', errors='coerce')
    creat_dt = pd.to_datetime(df['creatDate_str'], format='%Y%m%d', errors='coerce')
    df['car_age_days'] = (creat_dt - reg_dt).dt.days

    # 车龄（年）
    df['car_age_year'] = df['car_age_days'] // 365

    # 每年行驶公里数
    car_age_years = df['car_age_days'] / 365.0
    df['km_per_year'] = df['kilometer'] / car_age_years.replace(0, np.nan)

    # 删除原始日期列及临时列
    df.drop(columns=['regDate', 'creatDate', 'regDate_str', 'creatDate_str'], inplace=True)

    # --- notRepairedDamage: '-' -> NaN -> 数值型 ---
    df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', np.nan)
    df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce')

    # --- fuelType / bodyType / gearbox: 众数填充 ---
    for col in ['fuelType', 'bodyType', 'gearbox']:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

    # --- power: 0 视为缺失，异常值截断到 600，再用中位数填充 ---
    df.loc[df['power'] == 0, 'power'] = np.nan
    df['power'] = df['power'].clip(upper=600)
    df['power'] = df['power'].fillna(df['power'].median())

    # --- power 分桶 ---
    df['power_bucket'] = df['power'].map(bucket_power)

    # --- kilometer 分桶（单位：万公里）---
    df['kilometer_bucket'] = df['kilometer'].map(bucket_kilometer)

    # --- model: 0.0 视为缺失 ---
    df.loc[df['model'] == 0, 'model'] = np.nan

    # --- 多项式特征（经 screen_poly_features.py 筛选出的 Top 2）---
    df['v_0_plus_v_12'] = df['v_0'] + df['v_12']
    df['v_5_x_v_12'] = df['v_5'] * df['v_12']

    return df


print("\n" + "=" * 60)
print("2. 特征工程")
print("=" * 60)

df_train_pp = preprocess(df_train)
df_test_pp = preprocess(df_test)

# --- brand 统计特征（仅从训练集计算）---
brand_stats = df_train_pp.groupby('brand')['price'].agg(
    brand_price_mean='mean',
    brand_price_median='median',
    brand_price_std='std',
    brand_price_min='min',
    brand_price_max='max',
    brand_price_count='count'
).reset_index()

df_train_pp = df_train_pp.merge(brand_stats, on='brand', how='left')
df_test_pp = df_test_pp.merge(brand_stats, on='brand', how='left')

# --- brand x model 分组聚合统计特征（仅从训练集计算）---
agg_cols = ['price', 'power', 'kilometer']
bm_stats = df_train_pp.groupby(['brand', 'model'])[agg_cols].agg(['mean', 'median', 'std'])
bm_stats.columns = ['_'.join(col) for col in bm_stats.columns]
bm_stats = bm_stats.reset_index()

print(f"\nbrand x model 聚合特征数: {len(bm_stats.columns) - 2}")
print(f"brand x model 组合数: {len(bm_stats)}")

df_train_pp = df_train_pp.merge(bm_stats, on=['brand', 'model'], how='left')
df_test_pp = df_test_pp.merge(bm_stats, on=['brand', 'model'], how='left')

# --- brand x car_age_year 分组聚合统计特征（仅从训练集计算）---
ba_agg_cols = ['price', 'power', 'kilometer']
ba_stats = df_train_pp.groupby(['brand', 'car_age_year'])[ba_agg_cols].agg(['mean', 'median', 'std'])
ba_stats.columns = ['brand_age_' + '_'.join(col) for col in ba_stats.columns]
ba_stats = ba_stats.reset_index()

print(f"\nbrand x car_age_year 聚合特征数: {len(ba_stats.columns) - 2}")
print(f"brand x car_age_year 组合数: {len(ba_stats)}")

df_train_pp = df_train_pp.merge(ba_stats, on=['brand', 'car_age_year'], how='left')
df_test_pp = df_test_pp.merge(ba_stats, on=['brand', 'car_age_year'], how='left')

# 特征 = 除 SaleID 和 price 外的全部列
feature_cols = [c for c in df_train_pp.columns if c not in ('SaleID', 'price')]
X = df_train_pp[feature_cols]
y = df_train_pp['price']
X_test = df_test_pp[feature_cols]

# power_bucket / kilometer_bucket 为分类特征，需要指定给 CatBoost
cat_features_names = ['power_bucket', 'kilometer_bucket']
cat_features_idx = [feature_cols.index(c) for c in cat_features_names if c in feature_cols]

print(f"特征数量: {len(feature_cols)}")
print(f"分类特征索引: {cat_features_idx} ({cat_features_names})")
print(f"特征列表: {feature_cols}")
print(f"训练样本数: {X.shape[0]}")
print(f"price 统计: mean={y.mean():.0f}, median={y.median():.0f}, std={y.std():.0f}")

# power_bucket / kilometer_bucket 分布
print("\n--- power_bucket 分布 ---")
print(X['power_bucket'].value_counts().sort_index())
print("\n--- kilometer_bucket 分布 ---")
print(X['kilometer_bucket'].value_counts().sort_index())

# 缺失值检查
print("\n--- 缺失值检查 ---")
for col in feature_cols:
    n_null = X[col].isna().sum()
    if n_null > 0:
        print(f"  {col}: {n_null} ({n_null/len(X)*100:.2f}%)")


# ========== 3. 训练/验证集划分 (15% 验证集) ==========
print("\n" + "=" * 60)
print("3. 训练/验证集划分 (15% 验证集)")
print("=" * 60)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42
)
print(f"训练子集: {X_tr.shape[0]} 条")
print(f"验证集:   {X_val.shape[0]} 条")


# ========== 4. 第一阶段：early_stopping 确定最优迭代数 ==========
print("\n" + "=" * 60)
print("4. 第一阶段: early_stopping 确定最优迭代数")
print("=" * 60)

train_pool = Pool(X_tr, y_tr, cat_features=cat_features_idx)
val_pool = Pool(X_val, y_val, cat_features=cat_features_idx)

model_es = CatBoostRegressor(
    iterations=50000,
    learning_rate=0.015,
    depth=7,
    l2_leaf_reg=3.0,
    random_seed=42,
    loss_function='MAE',
    eval_metric='MAE',
    od_type='Iter',
    od_wait=100,
    verbose=200,
    random_strength=0.8,
    bagging_temperature=0.8,
    border_count=254,
    rsm=0.8
)

model_es.fit(train_pool, eval_set=val_pool)

best_round = model_es.best_iteration_
print(f"\nbest_iteration = {best_round}")

y_pred_val = model_es.predict(X_val)
y_pred_val = np.maximum(y_pred_val, 1.0)
mae_val = mean_absolute_error(y_val, y_pred_val)
print(f"验证集 MAE (early_stopping 模型): {mae_val:.2f}")


# ========== 5. 第二阶段：全量训练集重训 ==========
print("\n" + "=" * 60)
print(f"5. 第二阶段: 全量训练集重训 (iterations={best_round})")
print("=" * 60)

full_pool = Pool(X, y, cat_features=cat_features_idx)

model_final = CatBoostRegressor(
    iterations=best_round,
    learning_rate=0.015,
    depth=7,
    l2_leaf_reg=3.0,
    random_seed=42,
    loss_function='MAE',
    eval_metric='MAE',
    random_strength=0.8,
    bagging_temperature=0.8,
    border_count=254,
    rsm=0.8,
    verbose=False,
)

model_final.fit(full_pool)
print("全量训练完成")

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_final.get_feature_importance()
}).sort_values('importance', ascending=False)

print("\n--- 特征重要性 Top 20 ---")
print(importance.head(20).to_string(index=False))


# ========== 6. 测试集预测 & 输出 ==========
print("\n" + "=" * 60)
print("6. 测试集预测 & 输出")
print("=" * 60)

test_pool = Pool(X_test, cat_features=cat_features_idx)
pred = model_final.predict(test_pool)
pred = np.maximum(pred, 1.0)
print(f"预测价格范围: [{pred.min():.1f}, {pred.max():.1f}]")
print(f"预测均值: {pred.mean():.0f}")

submit = pd.DataFrame({
    'SaleID': test_SaleID,
    'price': pred
})

submit.to_csv('cb_submit_predictions.csv', index=False, encoding='utf-8')
print(f"\n已保存: cb_submit_predictions.csv ({len(submit)} 条)")
print(f"表头: {list(submit.columns)}")

# ========== 完成 ==========
print("\n" + "=" * 60)
print("建模完成!")
print(f"  best_iteration: {best_round}")
print(f"  验证集 MAE:     {mae_val:.2f}")
print(f"  输出文件:       cb_submit_predictions.csv")
print("=" * 60)
