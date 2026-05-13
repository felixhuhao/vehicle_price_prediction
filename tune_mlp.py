"""
MLP 单独调参脚本
  - 复用 v4 的特征工程，只训练 MLP
  - 不训练树模型，调参速度快
  - 修改下方 CONFIG 区域即可调参
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== CONFIG ====================
HIDDEN_DIMS = [1024, 512, 256]    # 网络结构
DROPOUT = 0.2               # Dropout
LR = 5e-4                   # 学习率
WEIGHT_DECAY = 1e-4         # L2 正则
N_EPOCHS = 1000              # 最大训练轮数
BATCH_SIZE = 1024            # 批大小
PATIENCE = 100               # 早停耐心
N_MLP_PER_FOLD = 5           # 每折训几个 MLP（种子不同，取平均）
N_FOLDS = 5
FOLD_SEEDS = [42, 123, 2024, 666, 999]
# ================================================

print(f"Device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"\nMLP Config:")
print(f"  hidden_dims={HIDDEN_DIMS}  dropout={DROPOUT}  lr={LR}")
print(f"  weight_decay={WEIGHT_DECAY}  epochs={N_EPOCHS}  batch={BATCH_SIZE}  patience={PATIENCE}")
print(f"  n_mlp_per_fold={N_MLP_PER_FOLD}")


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
print(f"训练集: {df_train.shape}  测试集: {df_test.shape}")


# ========== 2. 特征工程 ==========
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


print("\n特征工程...")
df_train_pp = preprocess(df_train)
df_test_pp = preprocess(df_test)


# ========== 3. 折内聚合 ==========
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
    return cols


# ========== 4. MLP 定义 ==========
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT):
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


def train_mlp(X_tr, y_tr, X_val, y_val, seed):
    torch.manual_seed(seed)

    scaler = StandardScaler()
    X_tr_s = np.nan_to_num(scaler.fit_transform(X_tr), nan=0.0)
    X_val_s = np.nan_to_num(scaler.transform(X_val), nan=0.0)

    X_tr_t = torch.FloatTensor(X_tr_s).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    X_val_t = torch.FloatTensor(X_val_s).to(device)

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MLP(X_tr_s.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    criterion = nn.L1Loss()

    best_val_mae = float('inf')
    best_state = None
    wait = 0

    for epoch in range(N_EPOCHS):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

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
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).cpu().numpy()
    return model, scaler, np.maximum(val_pred, 1.0), best_val_mae, epoch + 1


def predict_mlp(model, scaler, X):
    X_s = np.nan_to_num(scaler.transform(X), nan=0.0)
    X_t = torch.FloatTensor(X_s).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(X_t).cpu().numpy()
    return np.maximum(pred, 1.0)


# ========== 5. 5 折 CV ==========
print("\n" + "=" * 60)
print(f"2. 5 折 MLP CV")
print("=" * 60)

price_bins = pd.qcut(df_train_pp['price'], q=10, labels=False, duplicates='drop')
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_nn = np.zeros(len(df_train_pp))
test_nn = np.zeros(len(df_test_pp))

for fold, (tr_idx, val_idx) in enumerate(kf.split(df_train_pp, price_bins)):
    seed = FOLD_SEEDS[fold]
    print(f"\n--- Fold {fold + 1}/{N_FOLDS}  seed={seed} ---")

    df_tr = df_train_pp.iloc[tr_idx].copy()
    df_val = df_train_pp.iloc[val_idx].copy()
    df_te = df_test_pp.copy()

    df_tr, df_val_agg = compute_aggregations(df_tr, pd.concat([df_val, df_te], ignore_index=True))
    df_val = df_val_agg.iloc[:len(val_idx)].copy()
    df_te = df_val_agg.iloc[len(val_idx):].copy()

    feature_cols = select_features(df_tr)
    y_tr = df_tr['price'].values
    y_val = df_val['price'].values

    # label encode 分类特征
    X_tr_num = df_tr[feature_cols].copy()
    X_val_num = df_val[feature_cols].copy()
    X_te_num = df_te[feature_cols].copy()
    for col in cat_features_names:
        if col in feature_cols:
            combined = pd.concat([X_tr_num[col], X_val_num[col], X_te_num[col]], axis=0).astype('category')
            codes = combined.cat.codes
            n_tr = len(X_tr_num)
            n_val = len(X_val_num)
            X_tr_num[col] = codes.iloc[:n_tr].values
            X_val_num[col] = codes.iloc[n_tr:n_tr + n_val].values
            X_te_num[col] = codes.iloc[n_tr + n_val:].values

    # 多 MLP 种子平均
    val_preds = []
    test_preds = []
    for m in range(N_MLP_PER_FOLD):
        mlp_seed = seed * 1000 + m * 111
        model, scaler, val_pred, mae, epochs = train_mlp(
            X_tr_num.values, y_tr, X_val_num.values, y_val, seed=mlp_seed
        )
        val_preds.append(val_pred)
        test_preds.append(predict_mlp(model, scaler, X_te_num.values))
        print(f"  MLP-{m+1}/{N_MLP_PER_FOLD} seed={mlp_seed} MAE={mae:.1f} epochs={epochs}")

    val_avg = np.mean(val_preds, axis=0)
    test_avg = np.mean(test_preds, axis=0)
    oof_nn[val_idx] = val_avg
    test_nn += test_avg / N_FOLDS
    mae_avg = mean_absolute_error(y_val, np.maximum(val_avg, 1.0))
    print(f"  → 平均 MAE={mae_avg:.1f}  特征数={len(feature_cols)}")

oof_nn = np.maximum(oof_nn, 1.0)
test_nn = np.maximum(test_nn, 1.0)
y = df_train_pp['price'].values

oof_mae = mean_absolute_error(y, oof_nn)
print(f"\n{'=' * 60}")
print(f"MLP OOF MAE: {oof_mae:.2f}")
print(f"预测范围: [{test_nn.min():.1f}, {test_nn.max():.1f}]  均值: {test_nn.mean():.0f}")

# 输出测试集预测
submit = pd.DataFrame({'SaleID': test_SaleID, 'price': test_nn})
submit.to_csv('mlp_tune_submit.csv', index=False, encoding='utf-8')
print(f"已保存: mlp_tune_submit.csv")
print(f"Config: hidden_dims={HIDDEN_DIMS}  dropout={DROPOUT}  lr={LR}  wd={WEIGHT_DECAY}")
print("=" * 60)
