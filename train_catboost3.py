"""
二手车价格预测 - CatBoost 建模 v3
基于 v2，增加 log1p(price) 目标变换缓解长尾问题
流程：15% 验证集 + early_stopping -> best_iteration -> 全量训练集重训 -> 预测
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor, Pool
import warnings

warnings.filterwarnings('ignore')

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

    # --- model: 0.0 视为缺失 ---
    df.loc[df['model'] == 0, 'model'] = np.nan

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

# 特征 = 除 SaleID 和 price 外的全部列
feature_cols = [c for c in df_train_pp.columns if c not in ('SaleID', 'price')]
X = df_train_pp[feature_cols]
X_test = df_test_pp[feature_cols]

# --- 目标变换：log1p(price) ---
y_raw = df_train_pp['price']
y = np.log1p(y_raw)

print(f"特征数量: {len(feature_cols)}")
print(f"训练样本数: {X.shape[0]}")
print(f"price 原始:   mean={y_raw.mean():.0f}, median={y_raw.median():.0f}, std={y_raw.std():.0f}")
print(f"price 偏度:   {y_raw.skew():.4f}")
print(f"log1p(price): mean={y.mean():.4f}, median={y.median():.4f}, std={y.std():.4f}")
print(f"log1p(price) 偏度: {pd.Series(y).skew():.4f}")

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

train_pool = Pool(X_tr, y_tr)
val_pool = Pool(X_val, y_val)

model_es = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=3.0,
    random_seed=42,
    loss_function='RMSE',
    eval_metric='RMSE',
    od_type='Iter',
    od_wait=100,
    verbose=100,
    random_strength=0.5,
    bagging_temperature=0.8,
    border_count=128,
)

model_es.fit(train_pool, eval_set=val_pool)

best_round = model_es.best_iteration_
print(f"\nbest_iteration = {best_round}")

# 验证集评估（反变换回原始空间计算 MAE）
y_pred_val_log = model_es.predict(X_val)
y_pred_val = np.expm1(y_pred_val_log)
y_pred_val = np.maximum(y_pred_val, 1.0)
y_val_raw = np.expm1(y_val)
mae_val = mean_absolute_error(y_val_raw, y_pred_val)
print(f"验证集 MAE (反变换后): {mae_val:.2f}")


# ========== 5. 第二阶段：全量训练集重训 ==========
print("\n" + "=" * 60)
print(f"5. 第二阶段: 全量训练集重训 (iterations={best_round})")
print("=" * 60)

full_pool = Pool(X, y)

model_final = CatBoostRegressor(
    iterations=best_round,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=3.0,
    random_seed=42,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_strength=0.5,
    bagging_temperature=0.8,
    border_count=128,
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

pred_log = model_final.predict(X_test)
pred = np.expm1(pred_log)
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
print("  目标变换:       log1p(price)")
print(f"  best_iteration: {best_round}")
print(f"  验证集 MAE:     {mae_val:.2f}")
print("  输出文件:       cb_submit_predictions.csv")
print("=" * 60)
