"""
二手车价格预测 - 匿名特征多项式筛选
简版特征工程 + XGBoost 快速迭代，筛选 v_i * v_j 和 v_i + v_j 中的有效特征
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from itertools import combinations
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


# ========== 2. 简版特征工程 ==========
def preprocess(df):
    df = df.copy()

    df['regDate_str'] = df['regDate'].astype(str)
    df['reg_year'] = df['regDate_str'].str[:4].astype(int)
    df['reg_month'] = df['regDate_str'].str[4:6].astype(int)

    df['creatDate_str'] = df['creatDate'].astype(str)
    df['creat_year'] = df['creatDate_str'].str[:4].astype(int)
    df['creat_month'] = df['creatDate_str'].str[4:6].astype(int)

    reg_dt = pd.to_datetime(df['regDate_str'], format='%Y%m%d', errors='coerce')
    creat_dt = pd.to_datetime(df['creatDate_str'], format='%Y%m%d', errors='coerce')
    df['car_age_days'] = (creat_dt - reg_dt).dt.days

    df.drop(columns=['regDate', 'creatDate', 'regDate_str', 'creatDate_str'], inplace=True)

    df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', np.nan)
    df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce')

    df.loc[df['power'] == 0, 'power'] = np.nan
    df['power'] = df['power'].fillna(df['power'].median())

    df.loc[df['model'] == 0, 'model'] = np.nan

    return df


print("\n" + "=" * 60)
print("2. 简版特征工程")
print("=" * 60)

df_train_pp = preprocess(df_train)
df_test_pp = preprocess(df_test)

v_cols = [f'v_{i}' for i in range(15)]
base_cols = [c for c in df_train_pp.columns if c not in ('SaleID', 'price')]

print(f"基础特征数: {len(base_cols)}")


# ========== 3. 基线模型（无多项式特征）==========
print("\n" + "=" * 60)
print("3. 基线模型（无多项式特征）")
print("=" * 60)

X_base = df_train_pp[base_cols]
y = df_train_pp['price']

X_tr, X_val, y_tr, y_val = train_test_split(X_base, y, test_size=0.15, random_state=42)

model_base = xgb.XGBRegressor(
    n_estimators=3000, learning_rate=0.05, max_depth=7,
    min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42,
    n_jobs=-1, tree_method='hist', early_stopping_rounds=50,
)
model_base.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=200)

pred_base = model_base.predict(X_val)
pred_base = np.maximum(pred_base, 1.0)
mae_base = mean_absolute_error(y_val, pred_base)
print(f"\n基线 MAE: {mae_base:.2f}")


# ========== 4. 构造多项式特征 ==========
print("\n" + "=" * 60)
print("4. 构造多项式特征")
print("=" * 60)

v_pairs = list(combinations(v_cols, 2))
print(f"v 特征两两组合数: {len(v_pairs)} (乘法 + 加法 = {len(v_pairs) * 2})")

# 构造乘法特征
mul_features = pd.DataFrame(index=df_train_pp.index)
add_features = pd.DataFrame(index=df_train_pp.index)

for i, j in v_pairs:
    mul_features[f'{i}_x_{j}'] = df_train_pp[i] * df_train_pp[j]
    add_features[f'{i}_plus_{j}'] = df_train_pp[i] + df_train_pp[j]

print(f"乘法特征数: {mul_features.shape[1]}")
print(f"加法特征数: {add_features.shape[1]}")


# ========== 5. 相关性去冗余 ==========
print("\n" + "=" * 60)
print("5. 相关性去冗余（剔除与原始 V 特征相关性 > 0.95）")
print("=" * 60)

CORR_THRESHOLD = 0.95

def filter_by_corr(poly_df, v_data, threshold):
    kept = []
    dropped = []
    for col in poly_df.columns:
        corrs = [poly_df[col].corr(v_data[v_col]) for v_col in v_data.columns]
        max_corr = max(abs(c) for c in corrs)
        if max_corr > threshold:
            dropped.append((col, max_corr))
        else:
            kept.append(col)
    return kept, dropped

kept_mul, dropped_mul = filter_by_corr(mul_features, df_train_pp[v_cols], CORR_THRESHOLD)
kept_add, dropped_add = filter_by_corr(add_features, df_train_pp[v_cols], CORR_THRESHOLD)

print(f"\n乘法特征: 保留 {len(kept_mul)}/{len(v_pairs)}, 剔除 {len(dropped_mul)}")
print(f"加法特征: 保留 {len(kept_add)}/{len(v_pairs)}, 剔除 {len(dropped_add)}")
print(f"合计保留: {len(kept_mul) + len(kept_add)}")

# 展示被剔除的 top 10
print("\n--- 被剔除的乘法特征 Top 10（相关性最高）---")
dropped_mul_sorted = sorted(dropped_mul, key=lambda x: x[1], reverse=True)
for name, corr in dropped_mul_sorted[:10]:
    print(f"  {name}: {corr:.4f}")

print("\n--- 被剔除的加法特征 Top 10（相关性最高）---")
dropped_add_sorted = sorted(dropped_add, key=lambda x: x[1], reverse=True)
for name, corr in dropped_add_sorted[:10]:
    print(f"  {name}: {corr:.4f}")

# 合并保留的特征
poly_train = pd.concat([mul_features[kept_mul], add_features[kept_add]], axis=1)

print(f"\n多项式特征最终维度: {poly_train.shape}")


# ========== 6. 多项式特征筛选 ==========
print("\n" + "=" * 60)
print("6. 用 XGBoost 筛选多项式特征重要性")
print("=" * 60)

X_poly = pd.concat([df_train_pp[base_cols], poly_train], axis=1)

X_tr_p, X_val_p, y_tr_p, y_val_p = train_test_split(X_poly, y, test_size=0.15, random_state=42)

model_poly = xgb.XGBRegressor(
    n_estimators=5000, learning_rate=0.03, max_depth=7,
    min_child_weight=3, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42,
    n_jobs=-1, tree_method='hist', early_stopping_rounds=100,
)
model_poly.fit(X_tr_p, y_tr_p, eval_set=[(X_val_p, y_val_p)], verbose=200)

pred_poly = model_poly.predict(X_val_p)
pred_poly = np.maximum(pred_poly, 1.0)
mae_poly = mean_absolute_error(y_val_p, pred_poly)
print(f"\n多项式特征模型 MAE: {mae_poly:.2f}")
print(f"基线 MAE:          {mae_base:.2f}")
print(f"提升:              {mae_base - mae_poly:.2f}")


# ========== 7. 输出有价值的多项式特征 ==========
print("\n" + "=" * 60)
print("7. 多项式特征重要性排名（Top 30）")
print("=" * 60)

importance = pd.DataFrame({
    'feature': X_poly.columns,
    'importance': model_poly.feature_importances_
}).sort_values('importance', ascending=False)

# 仅看多项式特征
poly_importance = importance[importance['feature'].isin(poly_train.columns)].reset_index(drop=True)
print("\n--- 多项式特征重要性 Top 30 ---")
print(poly_importance.head(30).to_string(index=False))

# 重要性 > 0 的多项式特征
useful_poly = poly_importance[poly_importance['importance'] > 0].sort_values('importance', ascending=False)
print(f"\n重要性 > 0 的多项式特征: {len(useful_poly)}/{len(poly_train.columns)}")

# 区分乘法和加法
mul_useful = useful_poly[useful_poly['feature'].str.contains('_x_')]
add_useful = useful_poly[useful_poly['feature'].str.contains('_plus_')]
print(f"  乘法特征: {len(mul_useful)}")
print(f"  加法特征: {len(add_useful)}")

print("\n--- 有价值的乘法特征 ---")
print(mul_useful.head(20).to_string(index=False))

print("\n--- 有价值的加法特征 ---")
print(add_useful.head(20).to_string(index=False))


# ========== 8. 保存筛选结果 ==========
useful_list = useful_poly['feature'].tolist()
pd.DataFrame({'feature': useful_list}).to_csv('useful_poly_features.csv', index=False)
print(f"\n已保存筛选结果: useful_poly_features.csv ({len(useful_list)} 个特征)")

print("\n" + "=" * 60)
print("筛选完成!")
print(f"  基线 MAE:        {mae_base:.2f}")
print(f"  多项式 MAE:      {mae_poly:.2f}")
print(f"  提升:            {mae_base - mae_poly:.2f}")
print(f"  保留多项式特征:  {len(useful_list)}")
print("=" * 60)
