"""
二手车价格预测 - 多模型融合 v4
  - 在 v3 基础上增加 PyTorch MLP 神经网络
  - 三模型融合：CatBoost + LightGBM + MLP（去掉 XGBoost）
  - CV Target Encoding + StratifiedKFold + 每折不同种子
  - 加权平均融合
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# ========== 分桶函数 ==========
def bucket_power(power):
    if power <= 60: return '01_Micro_Electric'
    elif power <= 100: return '02_Economy'
    elif power <= 180: return '03_Best_Seller'
    elif power <= 300: return '04_Premium'
    elif power <= 500: return '05_Performance'
    else: return '06_Hypercar_Exotic'

def bucket_kilometer(km):
    if km < 0.1: return '01_Showroom'
    elif km < 1.0: return '02_Nearly_New'
    elif km < 3.0: return '03_Prime'
    elif km < 6.0: return '04_Normal'
    elif km < 10.0: return '05_Old'
    elif km < 20.0: return '06_High_Mileage'
    else: return '07_Scrap_or_RideHailing'

def bucket_car_age(age):
    if pd.isna(age): return '00_Unknown'
    if age <= 1: return '01_Nearly_New'
    elif age <= 3: return '02_Prime'
    elif age <= 5: return '03_Normal'
    elif age <= 8: return '04_Mature'
    elif age <= 12: return '05_Aging'
    else: return '06_Vintage'


# ========== 1. 加载数据 ==========
print("\n" + "=" * 60)
print("1. 加载数据")
print("=" * 60)

df_train = pd.read_csv('used_car_train.csv', sep=' ')
df_test = pd.read_csv('used_car_test.csv', sep=' ')
test_SaleID = df_test['SaleID'].values
print(f"训练集: {df_train.shape}")
print(f"测试集: {df_test.shape}")


# ========== 2. 基础特征工程 ==========
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


# ========== 3. 折内聚合函数 ==========
def compute_aggregations(df_tr, df_te):
    df_tr = df_tr.copy()
    df_te = df_te.copy()

    bs = df_tr.groupby('brand')['price'].agg(
        brand_price_mean='mean', brand_price_median='median', brand_price_std='std'
    ).reset_index()
    df_tr = df_tr.merge(bs, on='brand', how='left')
    df_te = df_te.merge(bs, on='brand', how='left')

    bd = df_tr.groupby('brand')['notRepairedDamage'].mean().reset_index()
    bd.columns = ['brand', 'brand_dmg_rate']
    df_tr = df_tr.merge(bd, on='brand', how='left')
    df_te = df_te.merge(bd, on='brand', how='left')

    ms = df_tr.groupby('model')['price'].agg(
        model_price_mean='mean', model_price_median='median', model_price_std='std'
    ).reset_index()
    df_tr = df_tr.merge(ms, on='model', how='left')
    df_te = df_te.merge(ms, on='model', how='left')

    bt = df_tr.groupby('bodyType')['price'].agg(
        body_price_mean='mean', body_price_median='median'
    ).reset_index()
    df_tr = df_tr.merge(bt, on='bodyType', how='left')
    df_te = df_te.merge(bt, on='bodyType', how='left')

    bg = df_tr.groupby(['brand', 'gearbox'])['price'].agg(
        brand_gear_price_mean='mean'
    ).reset_index()
    df_tr = df_tr.merge(bg, on=['brand', 'gearbox'], how='left')
    df_te = df_te.merge(bg, on=['brand', 'gearbox'], how='left')

    agg_cols = ['price', 'power', 'kilometer']
    bm = df_tr.groupby(['brand', 'model'])[agg_cols].agg(['mean', 'median', 'std'])
    bm.columns = ['_'.join(c) for c in bm.columns]
    bm = bm.reset_index()
    df_tr = df_tr.merge(bm, on=['brand', 'model'], how='left')
    df_te = df_te.merge(bm, on=['brand', 'model'], how='left')

    ba = df_tr.groupby(['brand', 'car_age_year'])[agg_cols].agg(['mean', 'median', 'std'])
    ba.columns = ['brand_age_' + '_'.join(c) for c in ba.columns]
    ba = ba.reset_index()
    df_tr = df_tr.merge(ba, on=['brand', 'car_age_year'], how='left')
    df_te = df_te.merge(ba, on=['brand', 'car_age_year'], how='left')

    return df_tr, df_te


drop_features = {
    'seller', 'offerType', 'creat_year', 'creat_month',
    'bodyType', 'gearbox', 'body_price_std', 'brand_gear_price_std',
    'brand_age_kilometer_median', 'brand_gear_price_median',
}
cat_features_names = ['power_bucket', 'kilometer_bucket', 'age_bucket']


def select_features(df):
    cols = [c for c in df.columns if c not in ('SaleID', 'price') and c not in drop_features]
    cat_idx = [cols.index(c) for c in cat_features_names if c in cols]
    return cols, cat_idx


# ========== MLP 定义 ==========
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[768, 384, 128], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_tr, y_tr, X_val, y_val, seed, n_epochs=1000, batch_size=1024, lr=5e-4):
    """训练 MLP，返回模型和最优 epoch 的预测"""
    torch.manual_seed(seed)

    # 标准化
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    # 填充 NaN 为 0（标准化后可能产生的）
    X_tr_s = np.nan_to_num(X_tr_s, nan=0.0)
    X_val_s = np.nan_to_num(X_val_s, nan=0.0)

    X_tr_t = torch.FloatTensor(X_tr_s).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    X_val_t = torch.FloatTensor(X_val_s).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLP(X_tr_s.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.L1Loss()  # MAE

    best_val_mae = float('inf')
    best_state = None
    patience = 100
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).cpu().numpy()
        val_mae = mean_absolute_error(y_val, np.maximum(val_pred, 1.0))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # 用最优模型预测
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).cpu().numpy()
    return model, scaler, np.maximum(val_pred, 1.0), best_val_mae


def predict_mlp(model, scaler, X):
    """用训练好的 MLP 预测"""
    X_s = scaler.transform(X)
    X_s = np.nan_to_num(X_s, nan=0.0)
    X_t = torch.FloatTensor(X_s).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(X_t).cpu().numpy()
    return np.maximum(pred, 1.0)


# ========== 4. 5 折 CV ==========
FOLD_SEEDS = [42, 123, 2024, 666, 999]
N_FOLDS = 5

print("\n" + "=" * 60)
print(f"3. 5 折 CV（三模型融合：CB + LGB + MLP）")
print(f"   折种子: {FOLD_SEEDS}")
print("=" * 60)

price_bins = pd.qcut(df_train_pp['price'], q=10, labels=False, duplicates='drop')
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_cb = np.zeros(len(df_train_pp))
oof_lgb = np.zeros(len(df_train_pp))
oof_nn = np.zeros(len(df_train_pp))
test_cb = np.zeros(len(df_test_pp))
test_lgb = np.zeros(len(df_test_pp))
test_nn = np.zeros(len(df_test_pp))

for fold, (tr_idx, val_idx) in enumerate(kf.split(df_train_pp, price_bins)):
    seed = FOLD_SEEDS[fold]
    print(f"\n--- Fold {fold + 1}/{N_FOLDS}  seed={seed} ---")

    df_tr = df_train_pp.iloc[tr_idx].copy()
    df_val = df_train_pp.iloc[val_idx].copy()
    df_te = df_test_pp.copy()

    # 折内聚合
    df_tr, df_val_agg = compute_aggregations(df_tr, pd.concat([df_val, df_te], ignore_index=True))
    df_val = df_val_agg.iloc[:len(val_idx)].copy()
    df_te = df_val_agg.iloc[len(val_idx):].copy()

    feature_cols, cat_idx = select_features(df_tr)
    y_tr = df_tr['price'].values
    y_val = df_val['price'].values

    X_tr = df_tr[feature_cols]
    X_val = df_val[feature_cols]
    X_te = df_te[feature_cols]

    # label encode for LGB/NN
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

    # --- LightGBM ---
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

    # --- MLP（3 个种子取平均）---
    N_MLP = 3
    val_preds = []
    test_preds = []
    for m in range(N_MLP):
        mlp_seed = seed * 1000 + m * 111
        mlp_model, mlp_scaler, val_pred, mae_nn = train_mlp(
            X_tr_num.values, y_tr, X_val_num.values, y_val, seed=mlp_seed
        )
        val_preds.append(val_pred)
        test_preds.append(predict_mlp(mlp_model, mlp_scaler, X_te_num.values))
        print(f"    MLP-{m+1}/{N_MLP} seed={mlp_seed} MAE={mae_nn:.1f}")

    nn_val_avg = np.mean(val_preds, axis=0)
    nn_test_avg = np.mean(test_preds, axis=0)
    oof_nn[val_idx] = nn_val_avg
    test_nn += nn_test_avg / N_FOLDS
    mae_nn_avg = mean_absolute_error(y_val, np.maximum(nn_val_avg, 1.0))

    print(f"  CB iter={cb.best_iteration_:5d} MAE={mae_cb:.1f}  "
          f"LGB iter={lgb.best_iteration_:5d} MAE={mae_lgb:.1f}  "
          f"MLP(avg) MAE={mae_nn_avg:.1f}  "
          f"特征数={len(feature_cols)}")

y = df_train_pp['price'].values

oof_cb = np.maximum(oof_cb, 1.0)
oof_lgb = np.maximum(oof_lgb, 1.0)
oof_nn = np.maximum(oof_nn, 1.0)
test_cb = np.maximum(test_cb, 1.0)
test_lgb = np.maximum(test_lgb, 1.0)
test_nn = np.maximum(test_nn, 1.0)

print(f"\n单模型 OOF MAE:")
print(f"  CatBoost:  {mean_absolute_error(y, oof_cb):.2f}")
print(f"  LightGBM:  {mean_absolute_error(y, oof_lgb):.2f}")
print(f"  MLP:       {mean_absolute_error(y, oof_nn):.2f}")


# ========== 5. 三模型加权融合 ==========
print("\n" + "=" * 60)
print("4. 三模型加权融合（CB + LGB + MLP）")
print("=" * 60)


def blend_mae(weights):
    w = np.maximum(np.array(weights), 0)
    w /= w.sum()
    pred = w[0] * oof_cb + w[1] * oof_lgb + w[2] * oof_nn
    return mean_absolute_error(y, pred)


# 网格搜索初始值
best_mae_w, best_w = float('inf'), [1/3, 1/3, 1/3]
for w0 in np.arange(0.1, 0.9, 0.05):
    for w1 in np.arange(0.1, 0.9 - w0, 0.05):
        w2 = 1.0 - w0 - w1
        if w2 < 0.05: continue
        mae = blend_mae([w0, w1, w2])
        if mae < best_mae_w:
            best_mae_w, best_w = mae, [w0, w1, w2]

res = minimize(blend_mae, best_w, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 0.001})
fw = np.maximum(res.x, 0); fw /= fw.sum()

pred_blend = np.maximum(fw[0] * test_cb + fw[1] * test_lgb + fw[2] * test_nn, 1.0)
submit_blend = pd.DataFrame({'SaleID': test_SaleID, 'price': pred_blend})
submit_blend.to_csv('ensemble_v4_weighted_submit.csv', index=False, encoding='utf-8')

print(f"\n[三模型加权平均]")
print(f"  权重: CB={fw[0]:.3f}  LGB={fw[1]:.3f}  MLP={fw[2]:.3f}")
print(f"  OOF MAE: {res.fun:.2f}")
print(f"  范围: [{pred_blend.min():.1f}, {pred_blend.max():.1f}]  均值: {pred_blend.mean():.0f}")
print(f"  已保存: ensemble_v4_weighted_submit.csv")


# ========== 完成 ==========
print("\n" + "=" * 60)
print("全部完成!")
print(f"  折种子: {FOLD_SEEDS}")
print(f"  特征数: {len(feature_cols)}")
print(f"  单模型 OOF MAE:")
print(f"    CB={mean_absolute_error(y, oof_cb):.2f}  LGB={mean_absolute_error(y, oof_lgb):.2f}  "
      f"MLP={mean_absolute_error(y, oof_nn):.2f}")
print(f"  三模型加权 OOF MAE: {res.fun:.2f}")
print(f"  输出: ensemble_v4_weighted_submit.csv")
print("=" * 60)
