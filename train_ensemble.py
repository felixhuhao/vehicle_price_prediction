"""
二手车价格预测 - 多模型融合 v3
  - CV Target Encoding（聚合特征在 fold 内计算，无泄露）
  - 每折不同种子（种子多样性 + 折内编码，一次循环完成）
  - Stacking（Ridge 第二层）+ 加权平均双输出
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


# ========== 分桶函数 ==========
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

def bucket_car_age(age):
    if pd.isna(age):
        return '00_Unknown'
    if age <= 1:
        return '01_Nearly_New'
    elif age <= 3:
        return '02_Prime'
    elif age <= 5:
        return '03_Normal'
    elif age <= 8:
        return '04_Mature'
    elif age <= 12:
        return '05_Aging'
    else:
        return '06_Vintage'


# ========== 1. 加载数据 ==========
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

df_train = pd.read_csv('used_car_train.csv', sep=' ')
df_test = pd.read_csv('used_car_test.csv', sep=' ')
test_SaleID = df_test['SaleID'].values
print(f"训练集: {df_train.shape}")
print(f"测试集: {df_test.shape}")


# ========== 2. 基础特征工程（不含聚合）==========
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
    df['car_age_year'] = df['car_age_days'] // 365

    car_age_years = df['car_age_days'] / 365.0
    df['km_per_year'] = df['kilometer'] / car_age_years.replace(0, np.nan)

    df.drop(columns=['regDate', 'creatDate', 'regDate_str', 'creatDate_str'], inplace=True)

    df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', np.nan)
    df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce')

    for col in ['fuelType', 'bodyType', 'gearbox']:
        df[col] = df[col].fillna(df[col].mode()[0])

    df.loc[df['power'] == 0, 'power'] = np.nan
    df['power'] = df['power'].clip(upper=600)
    df['power'] = df['power'].fillna(df['power'].median())

    df['power_bucket'] = df['power'].map(bucket_power)
    df['kilometer_bucket'] = df['kilometer'].map(bucket_kilometer)

    df.loc[df['model'] == 0, 'model'] = np.nan

    df['v_0_plus_v_12'] = df['v_0'] + df['v_12']
    df['v_5_x_v_12'] = df['v_5'] * df['v_12']
    df['v_3_x_v_8'] = df['v_3'] * df['v_8']
    df['v_3_x_v_12'] = df['v_3'] * df['v_12']

    v_cols = [f'v_{i}' for i in range(15)]
    df['v_row_std'] = df[v_cols].std(axis=1)

    car_age_yr = df['car_age_year'].replace(0, np.nan)
    df['power_age_ratio'] = df['power'] / car_age_yr

    df['dmg_x_age'] = df['notRepairedDamage'] * df['car_age_year']
    df['age_bucket'] = df['car_age_year'].map(bucket_car_age)

    return df


print("\n" + "=" * 60)
print("2. 基础特征工程")
print("=" * 60)

df_train_pp = preprocess(df_train)
df_test_pp = preprocess(df_test)

# 保留 price 列用于 fold 内聚合（后面会从特征中移除）
# 保留 bodyType / gearbox 用于聚合 join（后面会从特征中移除）


# ========== 3. 折内聚合函数 ==========
def compute_aggregations(df_tr, df_te):
    """从训练子集计算聚合特征，映射到训练子集和测试集"""
    df_tr = df_tr.copy()
    df_te = df_te.copy()

    # brand 统计
    bs = df_tr.groupby('brand')['price'].agg(
        brand_price_mean='mean', brand_price_median='median', brand_price_std='std'
    ).reset_index()
    df_tr = df_tr.merge(bs, on='brand', how='left')
    df_te = df_te.merge(bs, on='brand', how='left')

    # brand 损坏率
    bd = df_tr.groupby('brand')['notRepairedDamage'].mean().reset_index()
    bd.columns = ['brand', 'brand_dmg_rate']
    df_tr = df_tr.merge(bd, on='brand', how='left')
    df_te = df_te.merge(bd, on='brand', how='left')

    # model 统计
    ms = df_tr.groupby('model')['price'].agg(
        model_price_mean='mean', model_price_median='median', model_price_std='std'
    ).reset_index()
    df_tr = df_tr.merge(ms, on='model', how='left')
    df_te = df_te.merge(ms, on='model', how='left')

    # bodyType 统计
    bt = df_tr.groupby('bodyType')['price'].agg(
        body_price_mean='mean', body_price_median='median'
    ).reset_index()
    df_tr = df_tr.merge(bt, on='bodyType', how='left')
    df_te = df_te.merge(bt, on='bodyType', how='left')

    # brand x gearbox
    bg = df_tr.groupby(['brand', 'gearbox'])['price'].agg(
        brand_gear_price_mean='mean'
    ).reset_index()
    df_tr = df_tr.merge(bg, on=['brand', 'gearbox'], how='left')
    df_te = df_te.merge(bg, on=['brand', 'gearbox'], how='left')

    # brand x model
    agg_cols = ['price', 'power', 'kilometer']
    bm = df_tr.groupby(['brand', 'model'])[agg_cols].agg(['mean', 'median', 'std'])
    bm.columns = ['_'.join(c) for c in bm.columns]
    bm = bm.reset_index()
    df_tr = df_tr.merge(bm, on=['brand', 'model'], how='left')
    df_te = df_te.merge(bm, on=['brand', 'model'], how='left')

    # brand x car_age_year
    ba = df_tr.groupby(['brand', 'car_age_year'])[agg_cols].agg(['mean', 'median', 'std'])
    ba.columns = ['brand_age_' + '_'.join(c) for c in ba.columns]
    ba = ba.reset_index()
    df_tr = df_tr.merge(ba, on=['brand', 'car_age_year'], how='left')
    df_te = df_te.merge(ba, on=['brand', 'car_age_year'], how='left')

    return df_tr, df_te


# 要过滤的低重要性特征
drop_features = {
    'seller', 'offerType', 'creat_year', 'creat_month',
    'bodyType', 'gearbox', 'body_price_std', 'brand_gear_price_std',
    'brand_age_kilometer_median', 'brand_gear_price_median',
}
cat_features_names = ['power_bucket', 'kilometer_bucket', 'age_bucket']


def select_features(df):
    """提取特征矩阵"""
    cols = [c for c in df.columns if c not in ('SaleID', 'price') and c not in drop_features]
    cat_idx = [cols.index(c) for c in cat_features_names if c in cols]
    return cols, cat_idx


# ========== 4. 5 折 CV（每折不同种子）==========
FOLD_SEEDS = [42, 123, 2024, 666, 999]
N_FOLDS = 5

print("\n" + "=" * 60)
print("3. 5 折 CV（每折独立种子 + CV Target Encoding）")
print(f"   折种子: {FOLD_SEEDS}")
print("=" * 60)

# 按 price 分箱做 StratifiedKFold，确保每折价格分布一致
price_bins = pd.qcut(df_train_pp['price'], q=10, labels=False, duplicates='drop')
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_cb = np.zeros(len(df_train_pp))
oof_lgb = np.zeros(len(df_train_pp))
oof_xgb = np.zeros(len(df_train_pp))
test_cb = np.zeros(len(df_test_pp))
test_lgb = np.zeros(len(df_test_pp))
test_xgb = np.zeros(len(df_test_pp))

for fold, (tr_idx, val_idx) in enumerate(kf.split(df_train_pp, price_bins)):
    seed = FOLD_SEEDS[fold]
    print(f"\n--- Fold {fold + 1}/{N_FOLDS}  seed={seed} ---")

    df_tr = df_train_pp.iloc[tr_idx].copy()
    df_val = df_train_pp.iloc[val_idx].copy()
    df_te = df_test_pp.copy()

    # 折内聚合（CV Target Encoding）
    df_tr, df_val_agg = compute_aggregations(df_tr, pd.concat([df_val, df_te], ignore_index=True))
    # 拆回 val 和 test
    df_val = df_val_agg.iloc[:len(val_idx)].copy()
    df_te = df_val_agg.iloc[len(val_idx):].copy()

    # 提取特征
    feature_cols, cat_idx = select_features(df_tr)
    y_tr = df_tr['price'].values
    y_val = df_val['price'].values

    X_tr = df_tr[feature_cols]
    X_val = df_val[feature_cols]
    X_te = df_te[feature_cols]

    # label encode for LGB/XGB
    X_tr_num = X_tr.copy()
    X_val_num = X_val.copy()
    X_te_num = X_te.copy()
    for col in cat_features_names:
        if col in feature_cols:
            combined = pd.concat([X_tr_num[col], X_val_num[col], X_te_num[col]], axis=0).astype('category')
            codes = combined.cat.codes
            n_tr = len(X_tr_num)
            n_val = len(X_val_num)
            X_tr_num[col] = codes.iloc[:n_tr].values
            X_val_num[col] = codes.iloc[n_tr:n_tr + n_val].values
            X_te_num[col] = codes.iloc[n_tr + n_val:].values

    lgb_cat_idx = [feature_cols.index(c) for c in cat_features_names if c in feature_cols]

    # --- CatBoost ---
    cb_train = Pool(X_tr, y_tr, cat_features=cat_idx)
    cb_val = Pool(X_val, y_val, cat_features=cat_idx)
    cb = CatBoostRegressor(
        iterations=30000, learning_rate=0.03, depth=7, l2_leaf_reg=3.0,
        random_seed=seed, loss_function='MAE', eval_metric='MAE',
        od_type='Iter', od_wait=100, random_strength=0.8,
        bagging_temperature=0.8, border_count=254, rsm=0.8, verbose=0,
    )
    cb.fit(cb_train, eval_set=cb_val)
    oof_cb[val_idx] = cb.predict(X_val)
    test_cb += cb.predict(Pool(X_te, cat_features=cat_idx)) / N_FOLDS
    mae_cb = mean_absolute_error(y_val, np.maximum(oof_cb[val_idx], 1.0))

    # --- LightGBM（log1p 目标变换）---
    y_tr_log = np.log1p(y_tr)
    y_val_log = np.log1p(y_val)
    lgb = LGBMRegressor(
        n_estimators=30000, learning_rate=0.03, max_depth=7, num_leaves=127,
        reg_lambda=3.0, reg_alpha=0.5, min_child_samples=30,
        random_state=seed, objective='mae', metric='mae', verbose=-1,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
    )
    lgb.fit(
        X_tr_num, y_tr_log, eval_set=[(X_val_num, y_val_log)],
        callbacks=[
            __import__('lightgbm').early_stopping(100, verbose=False),
            __import__('lightgbm').log_evaluation(0),
        ],
        categorical_feature=lgb_cat_idx,
    )
    oof_lgb[val_idx] = np.expm1(lgb.predict(X_val_num))
    test_lgb += np.expm1(lgb.predict(X_te_num)) / N_FOLDS
    mae_lgb = mean_absolute_error(y_val, np.maximum(oof_lgb[val_idx], 1.0))

    # --- XGBoost ---
    xgb = XGBRegressor(
        n_estimators=30000, learning_rate=0.03, max_depth=7,
        reg_lambda=3.0, reg_alpha=0.5, min_child_weight=10, gamma=0.1,
        random_state=seed, objective='reg:absoluteerror', eval_metric='mae',
        early_stopping_rounds=100, verbosity=0,
        colsample_bytree=0.8, subsample=0.8,
    )
    xgb.fit(X_tr_num, y_tr, eval_set=[(X_val_num, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb.predict(X_val_num)
    test_xgb += xgb.predict(X_te_num) / N_FOLDS
    mae_xgb = mean_absolute_error(y_val, np.maximum(oof_xgb[val_idx], 1.0))

    print(f"  CB iter={cb.best_iteration_:5d} MAE={mae_cb:.1f}  "
          f"LGB iter={lgb.best_iteration_:5d} MAE={mae_lgb:.1f}  "
          f"XGB iter={xgb.best_iteration:5d} MAE={mae_xgb:.1f}  "
          f"特征数={len(feature_cols)}")

y = df_train_pp['price'].values

oof_cb = np.maximum(oof_cb, 1.0)
oof_lgb = np.maximum(oof_lgb, 1.0)
oof_xgb = np.maximum(oof_xgb, 1.0)
test_cb = np.maximum(test_cb, 1.0)
test_lgb = np.maximum(test_lgb, 1.0)
test_xgb = np.maximum(test_xgb, 1.0)

print("\n单模型 OOF MAE:")
print(f"  CatBoost:  {mean_absolute_error(y, oof_cb):.2f}")
print(f"  LightGBM:  {mean_absolute_error(y, oof_lgb):.2f}")
print(f"  XGBoost:   {mean_absolute_error(y, oof_xgb):.2f}")


# ========== 5. Stacking（Ridge）==========
print("\n" + "=" * 60)
print("4. Stacking（Ridge 第二层）")
print("=" * 60)

X_stack_train = np.column_stack([oof_cb, oof_lgb, oof_xgb])
X_stack_test = np.column_stack([test_cb, test_lgb, test_xgb])

best_alpha = 1.0
best_stack_mae = float('inf')
for alpha in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_stack_train, y)
    mae = mean_absolute_error(y, np.maximum(ridge.predict(X_stack_train), 1.0))
    print(f"  alpha={alpha:6.2f}  MAE={mae:.2f}  coef={ridge.coef_}  intercept={ridge.intercept_:.1f}")
    if mae < best_stack_mae:
        best_stack_mae = mae
        best_alpha = alpha

ridge_final = Ridge(alpha=best_alpha, random_state=42)
ridge_final.fit(X_stack_train, y)
stack_mae = mean_absolute_error(y, np.maximum(ridge_final.predict(X_stack_train), 1.0))
print(f"\nRidge 最优 alpha={best_alpha}")
print(f"  系数: CB={ridge_final.coef_[0]:.4f}  LGB={ridge_final.coef_[1]:.4f}  XGB={ridge_final.coef_[2]:.4f}")
print(f"  截距: {ridge_final.intercept_:.1f}")
print(f"  Stacking OOF MAE: {stack_mae:.2f}")


# ========== 6. 输出 ==========
print("\n" + "=" * 60)
print("5. 预测 & 输出")
print("=" * 60)

# --- Stacking ---
pred_stack = np.maximum(ridge_final.predict(X_stack_test), 1.0)
submit_stack = pd.DataFrame({'SaleID': test_SaleID, 'price': pred_stack})
submit_stack.to_csv('ensemble_stacking_submit.csv', index=False, encoding='utf-8')
print(f"\n[Stacking] 范围: [{pred_stack.min():.1f}, {pred_stack.max():.1f}]  均值: {pred_stack.mean():.0f}")
print("  已保存: ensemble_stacking_submit.csv")

# --- 加权平均 ---
def blend_mae(weights):
    w = np.maximum(np.array(weights), 0)
    w /= w.sum()
    return mean_absolute_error(y, w[0] * oof_cb + w[1] * oof_lgb + w[2] * oof_xgb)

best_mae_w, best_w = float('inf'), [1/3, 1/3, 1/3]
for w0 in np.arange(0.1, 0.9, 0.05):
    for w1 in np.arange(0.1, 0.9 - w0, 0.05):
        w2 = 1.0 - w0 - w1
        if w2 < 0.05:
            continue
        mae = blend_mae([w0, w1, w2])
        if mae < best_mae_w:
            best_mae_w, best_w = mae, [w0, w1, w2]

res = minimize(blend_mae, best_w, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 0.001})
fw = np.maximum(res.x, 0)
fw /= fw.sum()

pred_blend = np.maximum(fw[0] * test_cb + fw[1] * test_lgb + fw[2] * test_xgb, 1.0)
submit_blend = pd.DataFrame({'SaleID': test_SaleID, 'price': pred_blend})
submit_blend.to_csv('ensemble_weighted_submit.csv', index=False, encoding='utf-8')
print(f"\n[加权平均] 权重: CB={fw[0]:.3f}  LGB={fw[1]:.3f}  XGB={fw[2]:.3f}")
print(f"  OOF MAE: {res.fun:.2f}")
print(f"  范围: [{pred_blend.min():.1f}, {pred_blend.max():.1f}]  均值: {pred_blend.mean():.0f}")
print("  已保存: ensemble_weighted_submit.csv")


# ========== 完成 ==========
print("\n" + "=" * 60)
print("全部完成!")
print(f"  折种子: {FOLD_SEEDS}")
print(f"  特征数: {len(feature_cols)}  (含 CV Target Encoding)")
print(f"  单模型 OOF MAE:  CB={mean_absolute_error(y, oof_cb):.2f}  LGB={mean_absolute_error(y, oof_lgb):.2f}  XGB={mean_absolute_error(y, oof_xgb):.2f}")
print(f"  Stacking OOF MAE:  {stack_mae:.2f}")
print(f"  加权平均 OOF MAE:  {res.fun:.2f}")
print("  输出: ensemble_stacking_submit.csv / ensemble_weighted_submit.csv")
print("=" * 60)
