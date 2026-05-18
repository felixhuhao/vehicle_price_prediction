"""
MLP training with out-of-fold target encoding.

This script is based on tune_mlp.py, but changes the price-based aggregate
features so training rows do not see their own target value.

Training cost is kept the same:
  - 5 outer folds
  - 5 MLPs per fold
  - same epochs, batch size, and patience

The extra inner folds only build features.
"""
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== CONFIG ====================
WEIGHT_DECAY = 1e-4
N_EPOCHS = 1000
BATCH_SIZE = 1024
PATIENCE = 100
N_FOLDS = 5
FOLD_SEEDS = [42, 123, 2024, 666, 999]

MLP_CONFIGS = [
    {
        "name": "base_1024_512_256",
        "hidden_dims": [1024, 512, 256],
        "dropout": 0.20,
        "lr": 5e-4,
        "seed_offset": 0,
    },
    {
        "name": "small_768_384_128",
        "hidden_dims": [768, 384, 128],
        "dropout": 0.20,
        "lr": 5e-4,
        "seed_offset": 111,
    },
    {
        "name": "wide_1280_640_320",
        "hidden_dims": [1280, 640, 320],
        "dropout": 0.25,
        "lr": 3e-4,
        "seed_offset": 222,
    },
    {
        "name": "lowdrop_1024_512_256",
        "hidden_dims": [1024, 512, 256],
        "dropout": 0.10,
        "lr": 3e-4,
        "seed_offset": 333,
    },
    {
        "name": "mid_1024_768_256",
        "hidden_dims": [1024, 768, 256],
        "dropout": 0.25,
        "lr": 5e-4,
        "seed_offset": 444,
    },
]

INNER_TE_FOLDS = 5
TE_ALPHA = 50.0
MIN_STD_COUNT = 5
ADD_COUNT_FEATURES = True
# ================================================


def bucket_power(power):
    if power <= 60:
        return "01_Micro_Electric"
    if power <= 100:
        return "02_Economy"
    if power <= 180:
        return "03_Best_Seller"
    if power <= 300:
        return "04_Premium"
    if power <= 500:
        return "05_Performance"
    return "06_Hypercar_Exotic"


def bucket_kilometer(km):
    if km < 0.1:
        return "01_Showroom"
    if km < 1.0:
        return "02_Nearly_New"
    if km < 3.0:
        return "03_Prime"
    if km < 6.0:
        return "04_Normal"
    if km < 10.0:
        return "05_Old"
    if km < 20.0:
        return "06_High_Mileage"
    return "07_Scrap_or_RideHailing"


def bucket_car_age(age):
    if pd.isna(age):
        return "00_Unknown"
    if age <= 1:
        return "01_Nearly_New"
    if age <= 3:
        return "02_Prime"
    if age <= 5:
        return "03_Normal"
    if age <= 8:
        return "04_Mature"
    if age <= 12:
        return "05_Aging"
    return "06_Vintage"


def preprocess(df):
    df = df.copy()

    df["regDate_str"] = df["regDate"].astype(str)
    df["reg_year"] = df["regDate_str"].str[:4].astype(int)
    df["reg_month"] = df["regDate_str"].str[4:6].astype(int)
    df["creatDate_str"] = df["creatDate"].astype(str)
    df["creat_year"] = df["creatDate_str"].str[:4].astype(int)
    df["creat_month"] = df["creatDate_str"].str[4:6].astype(int)

    reg_dt = pd.to_datetime(df["regDate_str"], format="%Y%m%d", errors="coerce")
    creat_dt = pd.to_datetime(df["creatDate_str"], format="%Y%m%d", errors="coerce")
    df["car_age_days"] = (creat_dt - reg_dt).dt.days
    df["car_age_year"] = df["car_age_days"] // 365
    car_age_years = df["car_age_days"] / 365.0
    df["km_per_year"] = df["kilometer"] / car_age_years.replace(0, np.nan)
    df.drop(columns=["regDate", "creatDate", "regDate_str", "creatDate_str"], inplace=True)

    df["notRepairedDamage"] = df["notRepairedDamage"].replace("-", np.nan)
    df["notRepairedDamage"] = pd.to_numeric(df["notRepairedDamage"], errors="coerce")
    for col in ["fuelType", "bodyType", "gearbox"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    df.loc[df["power"] == 0, "power"] = np.nan
    df["power"] = df["power"].clip(upper=600)
    df["power"] = df["power"].fillna(df["power"].median())
    df["power_bucket"] = df["power"].map(bucket_power)
    df["kilometer_bucket"] = df["kilometer"].map(bucket_kilometer)
    df.loc[df["model"] == 0, "model"] = np.nan

    df["v_0_plus_v_12"] = df["v_0"] + df["v_12"]
    df["v_5_x_v_12"] = df["v_5"] * df["v_12"]
    df["v_3_x_v_8"] = df["v_3"] * df["v_8"]
    df["v_3_x_v_12"] = df["v_3"] * df["v_12"]

    v_cols = [f"v_{i}" for i in range(15)]
    df["v_row_std"] = df[v_cols].std(axis=1)
    car_age_yr = df["car_age_year"].replace(0, np.nan)
    df["power_age_ratio"] = df["power"] / car_age_yr
    df["dmg_x_age"] = df["notRepairedDamage"] * df["car_age_year"]
    df["age_bucket"] = df["car_age_year"].map(bucket_car_age)
    return df


def _target_stat_columns(prefix, include_median=True, include_std=True):
    cols = [f"{prefix}_mean"]
    if include_median:
        cols.append(f"{prefix}_median")
    if include_std:
        cols.append(f"{prefix}_std")
    if ADD_COUNT_FEATURES:
        cols.extend([f"{prefix}_count", f"{prefix}_log_count"])
    return cols


def _fit_target_stats(source, keys, prefix, include_median, include_std, global_stats):
    grouped = source.groupby(keys, dropna=True)["price"].agg(["sum", "count", "median", "std"])
    grouped = grouped.reset_index()
    grouped[f"{prefix}_mean"] = (
        grouped["sum"] + global_stats["mean"] * TE_ALPHA
    ) / (grouped["count"] + TE_ALPHA)

    cols = list(keys) + [f"{prefix}_mean"]
    if include_median:
        grouped[f"{prefix}_median"] = grouped["median"]
        cols.append(f"{prefix}_median")
    if include_std:
        grouped[f"{prefix}_std"] = grouped["std"]
        grouped.loc[grouped["count"] < MIN_STD_COUNT, f"{prefix}_std"] = global_stats["std"]
        grouped[f"{prefix}_std"] = grouped[f"{prefix}_std"].fillna(global_stats["std"])
        cols.append(f"{prefix}_std")
    if ADD_COUNT_FEATURES:
        grouped[f"{prefix}_count"] = grouped["count"].astype(float)
        grouped[f"{prefix}_log_count"] = np.log1p(grouped["count"].astype(float))
        cols.extend([f"{prefix}_count", f"{prefix}_log_count"])
    return grouped[cols]


def _fill_target_stat_defaults(df, prefix, include_median, include_std, global_stats):
    df[f"{prefix}_mean"] = df[f"{prefix}_mean"].fillna(global_stats["mean"])
    if include_median:
        df[f"{prefix}_median"] = df[f"{prefix}_median"].fillna(global_stats["median"])
    if include_std:
        df[f"{prefix}_std"] = df[f"{prefix}_std"].fillna(global_stats["std"])
    if ADD_COUNT_FEATURES:
        df[f"{prefix}_count"] = df[f"{prefix}_count"].fillna(0.0)
        df[f"{prefix}_log_count"] = df[f"{prefix}_log_count"].fillna(0.0)
    return df


def _merge_target_stats(target, stats, keys, prefix, include_median, include_std, global_stats):
    target = target.merge(stats, on=keys, how="left")
    return _fill_target_stat_defaults(target, prefix, include_median, include_std, global_stats)


def add_oof_target_stats(df_tr, df_eval, keys, prefix, include_median=True, include_std=True, seed=42):
    df_tr = df_tr.copy()
    df_eval = df_eval.copy()
    global_stats = {
        "mean": df_tr["price"].mean(),
        "median": df_tr["price"].median(),
        "std": df_tr["price"].std(),
    }

    stat_cols = _target_stat_columns(prefix, include_median, include_std)
    for col in stat_cols:
        df_tr[col] = np.nan

    price_bins = pd.qcut(df_tr["price"], q=10, labels=False, duplicates="drop")
    inner_kf = StratifiedKFold(n_splits=INNER_TE_FOLDS, shuffle=True, random_state=seed)

    for inner_tr_idx, inner_val_idx in inner_kf.split(df_tr, price_bins):
        inner_source = df_tr.iloc[inner_tr_idx]
        inner_target = df_tr.iloc[inner_val_idx][list(keys)].copy()
        inner_target["_row_id"] = df_tr.index[inner_val_idx]

        stats = _fit_target_stats(
            inner_source, keys, prefix, include_median, include_std, global_stats
        )
        encoded = inner_target.merge(stats, on=keys, how="left")
        encoded = _fill_target_stat_defaults(
            encoded, prefix, include_median, include_std, global_stats
        )

        row_ids = encoded["_row_id"].values
        for col in stat_cols:
            df_tr.loc[row_ids, col] = encoded[col].values

    stats_full = _fit_target_stats(df_tr, keys, prefix, include_median, include_std, global_stats)
    df_eval = _merge_target_stats(
        df_eval, stats_full, keys, prefix, include_median, include_std, global_stats
    )
    return df_tr, df_eval


def add_non_target_aggregations(df_tr, df_eval):
    df_tr = df_tr.copy()
    df_eval = df_eval.copy()

    brand_dmg = df_tr.groupby("brand")["notRepairedDamage"].mean().reset_index()
    brand_dmg.columns = ["brand", "brand_dmg_rate"]
    df_tr = df_tr.merge(brand_dmg, on="brand", how="left")
    df_eval = df_eval.merge(brand_dmg, on="brand", how="left")

    bm = df_tr.groupby(["brand", "model"])[["power", "kilometer"]].agg(["mean", "median", "std"])
    bm.columns = ["_".join(c) for c in bm.columns]
    bm = bm.reset_index()
    df_tr = df_tr.merge(bm, on=["brand", "model"], how="left")
    df_eval = df_eval.merge(bm, on=["brand", "model"], how="left")

    ba = df_tr.groupby(["brand", "car_age_year"])[["power", "kilometer"]].agg(
        ["mean", "median", "std"]
    )
    ba.columns = ["brand_age_" + "_".join(c) for c in ba.columns]
    ba = ba.reset_index()
    df_tr = df_tr.merge(ba, on=["brand", "car_age_year"], how="left")
    df_eval = df_eval.merge(ba, on=["brand", "car_age_year"], how="left")
    return df_tr, df_eval


def compute_aggregations_oof(df_tr, df_eval, seed):
    df_tr = df_tr.copy()
    df_eval = df_eval.copy()
    original_index = df_tr.index
    df_tr.index = np.arange(len(df_tr))

    target_specs = [
        (["brand"], "brand_price", True, True),
        (["model"], "model_price", True, True),
        (["bodyType"], "body_price", True, False),
        (["brand", "gearbox"], "brand_gear_price", False, False),
        (["brand", "model"], "price", True, True),
        (["brand", "car_age_year"], "brand_age_price", True, True),
    ]
    for keys, prefix, include_median, include_std in target_specs:
        df_tr, df_eval = add_oof_target_stats(
            df_tr,
            df_eval,
            keys,
            prefix,
            include_median=include_median,
            include_std=include_std,
            seed=seed,
        )

    df_tr, df_eval = add_non_target_aggregations(df_tr, df_eval)
    df_tr.index = original_index
    return df_tr, df_eval


drop_features = {
    "seller",
    "offerType",
    "creat_year",
    "creat_month",
    "bodyType",
    "gearbox",
    "body_price_std",
    "brand_gear_price_std",
    "brand_age_kilometer_median",
    "brand_gear_price_median",
}
cat_features_names = ["power_bucket", "kilometer_bucket", "age_bucket"]


def select_features(df):
    return [c for c in df.columns if c not in ("SaleID", "price") and c not in drop_features]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
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


def train_mlp(X_tr, y_tr, X_val, y_val, seed, config):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    scaler = StandardScaler()
    X_tr_s = np.nan_to_num(scaler.fit_transform(X_tr), nan=0.0)
    X_val_s = np.nan_to_num(scaler.transform(X_val), nan=0.0)

    X_tr_t = torch.FloatTensor(X_tr_s).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    X_val_t = torch.FloatTensor(X_val_s).to(device)

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = MLP(
        X_tr_s.shape[1],
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    criterion = nn.L1Loss()

    best_val_mae = float("inf")
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


def encode_categorical_features(X_tr, X_val, X_te, feature_cols):
    X_tr_num = X_tr.copy()
    X_val_num = X_val.copy()
    X_te_num = X_te.copy()
    for col in cat_features_names:
        if col in feature_cols:
            combined = pd.concat([X_tr_num[col], X_val_num[col], X_te_num[col]], axis=0).astype(
                "category"
            )
            codes = combined.cat.codes
            n_tr = len(X_tr_num)
            n_val = len(X_val_num)
            X_tr_num[col] = codes.iloc[:n_tr].values
            X_val_num[col] = codes.iloc[n_tr:n_tr + n_val].values
            X_te_num[col] = codes.iloc[n_tr + n_val:].values
    return X_tr_num, X_val_num, X_te_num


def blend_mae(y, oof_by_config, weights):
    pred = np.average(oof_by_config, axis=0, weights=weights)
    return mean_absolute_error(y, np.maximum(pred, 1.0))


def optimize_blend_weights(y, oof_by_config):
    n_configs = oof_by_config.shape[0]
    candidates = [np.ones(n_configs) / n_configs]
    for idx in range(n_configs):
        one_hot = np.zeros(n_configs)
        one_hot[idx] = 1.0
        candidates.append(one_hot)

    best_w = min(candidates, key=lambda w: blend_mae(y, oof_by_config, w))
    best_mae = blend_mae(y, oof_by_config, best_w)

    for step in [0.20, 0.10, 0.05, 0.02, 0.01, 0.005]:
        improved = True
        while improved:
            improved = False
            for src in range(n_configs):
                for dst in range(n_configs):
                    if src == dst or best_w[src] < step:
                        continue
                    trial = best_w.copy()
                    trial[src] -= step
                    trial[dst] += step
                    mae = blend_mae(y, oof_by_config, trial)
                    if mae + 1e-9 < best_mae:
                        best_w = trial
                        best_mae = mae
                        improved = True
    return best_w, best_mae


def main():
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("\nMLP Config:")
    print(
        f"  weight_decay={WEIGHT_DECAY}  epochs={N_EPOCHS}  "
        f"batch={BATCH_SIZE}  patience={PATIENCE}"
    )
    print(f"  n_mlp_configs={len(MLP_CONFIGS)}")
    for idx, config in enumerate(MLP_CONFIGS, start=1):
        print(
            f"    {idx}. {config['name']}: hidden={config['hidden_dims']}  "
            f"dropout={config['dropout']}  lr={config['lr']}"
        )
    print(
        f"  OOF target encoding: inner_folds={INNER_TE_FOLDS}  "
        f"alpha={TE_ALPHA}  count_features={ADD_COUNT_FEATURES}"
    )

    print("\n" + "=" * 60)
    print("1. Load data")
    print("=" * 60)
    df_train = pd.read_csv("used_car_train.csv", sep=" ")
    df_test = pd.read_csv("used_car_test.csv", sep=" ")
    test_sale_id = df_test["SaleID"].values
    print(f"train: {df_train.shape}  test: {df_test.shape}")

    print("\nFeature engineering...")
    df_train_pp = preprocess(df_train)
    df_test_pp = preprocess(df_test)

    print("\n" + "=" * 60)
    print("2. 5-fold MLP CV with OOF target encoding")
    print("=" * 60)
    price_bins = pd.qcut(df_train_pp["price"], q=10, labels=False, duplicates="drop")
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    n_configs = len(MLP_CONFIGS)
    oof_by_config = np.zeros((n_configs, len(df_train_pp)))
    test_by_config = np.zeros((n_configs, len(df_test_pp)))
    oof_simple = np.zeros(len(df_train_pp))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(df_train_pp, price_bins)):
        seed = FOLD_SEEDS[fold]
        print(f"\n--- Fold {fold + 1}/{N_FOLDS}  seed={seed} ---")

        df_tr = df_train_pp.iloc[tr_idx].copy()
        df_val = df_train_pp.iloc[val_idx].copy()
        df_te = df_test_pp.copy()

        df_eval = pd.concat([df_val, df_te], ignore_index=True)
        df_tr, df_eval = compute_aggregations_oof(df_tr, df_eval, seed=seed)
        df_val = df_eval.iloc[:len(val_idx)].copy()
        df_te = df_eval.iloc[len(val_idx):].copy()

        feature_cols = select_features(df_tr)
        y_tr = df_tr["price"].values
        y_val = df_val["price"].values

        X_tr_num, X_val_num, X_te_num = encode_categorical_features(
            df_tr[feature_cols],
            df_val[feature_cols],
            df_te[feature_cols],
            feature_cols,
        )

        val_preds = []
        for config_idx, config in enumerate(MLP_CONFIGS):
            mlp_seed = seed * 1000 + config["seed_offset"]
            model, scaler, val_pred, mae, epochs = train_mlp(
                X_tr_num.values,
                y_tr,
                X_val_num.values,
                y_val,
                seed=mlp_seed,
                config=config,
            )
            test_pred = predict_mlp(model, scaler, X_te_num.values)
            val_preds.append(val_pred)
            oof_by_config[config_idx, val_idx] = val_pred
            test_by_config[config_idx] += test_pred / N_FOLDS
            print(
                f"  MLP-{config_idx + 1}/{n_configs} {config['name']} "
                f"seed={mlp_seed} MAE={mae:.1f} epochs={epochs}"
            )

        val_avg = np.mean(val_preds, axis=0)
        oof_simple[val_idx] = val_avg
        mae_avg = mean_absolute_error(y_val, np.maximum(val_avg, 1.0))
        print(f"  -> average MAE={mae_avg:.1f}  features={len(feature_cols)}")

    oof_by_config = np.maximum(oof_by_config, 1.0)
    test_by_config = np.maximum(test_by_config, 1.0)
    oof_simple = np.maximum(oof_simple, 1.0)
    test_simple = np.maximum(test_by_config.mean(axis=0), 1.0)
    y = df_train_pp["price"].values
    simple_mae = mean_absolute_error(y, oof_simple)
    blend_weights, blend_oof_mae = optimize_blend_weights(y, oof_by_config)
    test_weighted = np.maximum(np.average(test_by_config, axis=0, weights=blend_weights), 1.0)

    print(f"\n{'=' * 60}")
    print("Per-config OOF MAE:")
    for config_idx, config in enumerate(MLP_CONFIGS):
        mae = mean_absolute_error(y, oof_by_config[config_idx])
        print(f"  {config['name']}: {mae:.2f}")
    print(f"Simple average OOF MAE: {simple_mae:.2f}")
    print(f"Weighted blend OOF MAE: {blend_oof_mae:.2f}")
    print("Blend weights:")
    for config, weight in zip(MLP_CONFIGS, blend_weights):
        print(f"  {config['name']}: {weight:.3f}")
    print(
        f"Simple prediction range: [{test_simple.min():.1f}, {test_simple.max():.1f}]  "
        f"mean={test_simple.mean():.0f}"
    )
    print(
        f"Weighted prediction range: [{test_weighted.min():.1f}, {test_weighted.max():.1f}]  "
        f"mean={test_weighted.mean():.0f}"
    )

    submit_simple = pd.DataFrame({"SaleID": test_sale_id, "price": test_simple})
    submit_simple.to_csv("mlp_tune_oof_te_multiconfig_avg_submit.csv", index=False, encoding="utf-8")
    submit_weighted = pd.DataFrame({"SaleID": test_sale_id, "price": test_weighted})
    submit_weighted.to_csv(
        "mlp_tune_oof_te_multiconfig_weighted_submit.csv",
        index=False,
        encoding="utf-8",
    )
    print("Saved: mlp_tune_oof_te_multiconfig_avg_submit.csv")
    print("Saved: mlp_tune_oof_te_multiconfig_weighted_submit.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
