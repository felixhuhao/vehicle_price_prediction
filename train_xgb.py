"""
二手车价格预测 - XGBoost 建模
流程：15% 验证集 + early_stopping -> best_iteration -> 全量训练集重训 -> 预测
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# ========== 1. 加载数据 ==========
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

df_train = pd.read_csv('used_car_train.csv', sep=' ')
df_test = pd.read_csv('used_car_test.csv', sep=' ')

# 保留原始 SaleID 用于输出
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

    # 每年行驶公里数 = kilometer / (车龄天数 / 365)
    car_age_years = df['car_age_days'] / 365.0
    df['km_per_year'] = df['kilometer'] / car_age_years.replace(0, np.nan)

    # 删除原始日期列及临时列，避免与衍生列重复
    df.drop(columns=['regDate', 'creatDate', 'regDate_str', 'creatDate_str'], inplace=True)

    # --- notRepairedDamage: '-' -> NaN -> numeric ---
    df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', np.nan)
    df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce')

    # --- power: 0 视为缺失，用中位数填充 ---
    df.loc[df['power'] == 0, 'power'] = np.nan
    df['power'] = df['power'].fillna(df['power'].median())

    # --- model: 0.0 视为缺失 ---
    df.loc[df['model'] == 0, 'model'] = np.nan

    return df


print("\n" + "=" * 60)
print("2. 特征工程")
print("=" * 60)

df_train_pp = preprocess(df_train)
df_test_pp = preprocess(df_test)

# --- brand 统计特征（仅从训练集计算，映射到全部数据）---
brand_stats = df_train_pp.groupby('brand')['price'].agg(
    brand_price_mean='mean',
    brand_price_median='median',
    brand_price_std='std',
    brand_price_min='min',
    brand_price_max='max',
    brand_price_count='count'
).reset_index()

print("\n--- brand 统计特征 ---")
print(brand_stats.head(10).to_string(index=False))

df_train_pp = df_train_pp.merge(brand_stats, on='brand', how='left')
df_test_pp = df_test_pp.merge(brand_stats, on='brand', how='left')

# 特征 = 除 SaleID 和 price 外的全部列
feature_cols = [c for c in df_train_pp.columns if c not in ('SaleID', 'price')]
X = df_train_pp[feature_cols]
y = df_train_pp['price']

X_test = df_test_pp[feature_cols]

print(f"\n特征数量: {len(feature_cols)}")
print(f"特征列表: {feature_cols}")
print(f"训练样本数: {X.shape[0]}")
print(f"price 统计: mean={y.mean():.0f}, median={y.median():.0f}, std={y.std():.0f}")


# ========== 3. 训练/验证集划分 (15% 验证集) ==========
print("\n" + "=" * 60)
print("3. 训练/验证集划分 (15% 验证集)")
print("=" * 60)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42
)
print(f"训练子集: {X_tr.shape[0]} 条")
print(f"验证集:   {X_val.shape[0]} 条")


# ========== 4. 第一阶段：early_stopping 确定最优树棵数 ==========
print("\n" + "=" * 60)
print("4. 第一阶段: early_stopping 确定最优树棵数")
print("=" * 60)

xgb_params = dict(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    early_stopping_rounds=50,
)

model_es = xgb.XGBRegressor(**xgb_params)
model_es.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)

best_round = model_es.best_iteration
print(f"\nbest_iteration = {best_round}")

# 验证集 MAE
y_pred_val = model_es.predict(X_val)
y_pred_val = np.maximum(y_pred_val, 1.0)
mae_val = mean_absolute_error(y_val, y_pred_val)
print(f"验证集 MAE (early_stopping 模型): {mae_val:.2f}")


# ========== 5. 第二阶段：全量训练集重训 ==========
print("\n" + "=" * 60)
print(f"5. 第二阶段: 全量训练集重训 (n_estimators={best_round})")
print("=" * 60)

model_final = xgb.XGBRegressor(
    n_estimators=best_round,
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
)
model_final.fit(X, y, verbose=False)
print("全量训练完成")

# 特征重要性 Top 20
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_final.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- 特征重要性 Top 20 ---")
print(importance.head(20).to_string(index=False))


# ========== 6. 测试集预测 & 输出 ==========
print("\n" + "=" * 60)
print("6. 测试集预测 & 输出")
print("=" * 60)

pred = model_final.predict(X_test)
pred = np.maximum(pred, 1.0)
print(f"预测价格范围: [{pred.min():.1f}, {pred.max():.1f}]")
print(f"预测均值: {pred.mean():.0f}")

submit = pd.DataFrame({
    'SaleID': test_SaleID,
    'price': pred
})

submit.to_csv('xgb_submit_predictions.csv', index=False, encoding='utf-8')
print(f"\n已保存: xgb_submit_predictions.csv ({len(submit)} 条)")
print(f"表头: {list(submit.columns)}")

# ========== 完成 ==========
print("\n" + "=" * 60)
print("建模完成!")
print(f"  best_iteration: {best_round}")
print(f"  验证集 MAE:     {mae_val:.2f}")
print(f"  输出文件:       xgb_submit_predictions.csv")
print("=" * 60)
