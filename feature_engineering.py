"""
二手车价格预测 - 特征工程：power 异常值验证与截断
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'feature_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 1. 加载数据 ==========
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

df_train = pd.read_csv('used_car_train.csv', sep=' ')
df_test = pd.read_csv('used_car_test.csv', sep=' ')

print(f"训练集: {df_train.shape}")
print(f"测试集: {df_test.shape}")


# ========== 2. power 异常值分析（原始数据）==========
print("\n" + "=" * 60)
print("2. power 异常值分析（原始数据）")
print("=" * 60)

for name, df in [('训练集', df_train), ('测试集', df_test)]:
    p = df['power']
    print(f"\n--- {name} ---")
    print(f"  总记录数:  {len(p)}")
    print(f"  均值:      {p.mean():.1f}")
    print(f"  中位数:    {p.median():.1f}")
    print(f"  标准差:    {p.std():.1f}")
    print(f"  最小值:    {p.min()}")
    print(f"  最大值:    {p.max()}")
    print("  分位数:")
    for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
        print(f"    {q*100:5.1f}%: {p.quantile(q):.0f}")
    print(f"  = 0 的记录数:     {(p == 0).sum()} ({(p == 0).mean()*100:.2f}%)")
    print(f"  > 600 的记录数:   {(p > 600).sum()} ({(p > 600).mean()*100:.2f}%)")
    print(f"  > 400 的记录数:   {(p > 400).sum()} ({(p > 400).mean()*100:.2f}%)")
    print(f"  > 300 的记录数:   {(p > 300).sum()} ({(p > 300).mean()*100:.2f}%)")


# ========== 3. 可视化 ==========
print("\n" + "=" * 60)
print("3. 可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 3.1 直方图（全量）
axes[0, 0].hist(df_train['power'], bins=200, color='steelblue', edgecolor='white')
axes[0, 0].axvline(x=600, color='red', linestyle='--', linewidth=1.5, label='600')
axes[0, 0].set_title('power 分布 (全量)')
axes[0, 0].set_xlabel('power')
axes[0, 0].set_ylabel('频数')
axes[0, 0].legend()

# 3.2 直方图（power <= 600 放大）
p_filtered = df_train['power'][df_train['power'] <= 600]
axes[0, 1].hist(p_filtered, bins=100, color='steelblue', edgecolor='white')
axes[0, 1].set_title('power 分布 (<= 600)')
axes[0, 1].set_xlabel('power')
axes[0, 1].set_ylabel('频数')

# 3.3 箱线图
bp = axes[1, 0].boxplot(df_train['power'], vert=True, patch_artist=True,
                        boxprops=dict(facecolor='steelblue', alpha=0.7))
axes[1, 0].set_title('power 箱线图')
axes[1, 0].set_ylabel('power')

# 3.4 power vs price 散点图（抽样）
sample = df_train.sample(n=min(10000, len(df_train)), random_state=42)
axes[1, 1].scatter(sample['power'], sample['price'], alpha=0.3, s=5, color='steelblue')
axes[1, 1].axvline(x=600, color='red', linestyle='--', linewidth=1.5, label='power=600')
axes[1, 1].set_xlabel('power')
axes[1, 1].set_ylabel('price')
axes[1, 1].set_title('power vs price')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/power_analysis.png', dpi=150)
plt.close()
print(f"图表已保存: {OUTPUT_DIR}/power_analysis.png")


# ========== 4. 异常值截断处理 ==========
print("\n" + "=" * 60)
print("4. 异常值截断处理")
print("=" * 60)


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

    # --- notRepairedDamage: '-' -> NaN -> 众数填充 ---
    df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', np.nan)
    df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce')
    mode_val = df['notRepairedDamage'].mode()[0]
    df['notRepairedDamage'] = df['notRepairedDamage'].fillna(mode_val)

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


df_train_pp = preprocess(df_train)
df_test_pp = preprocess(df_test)

# brand 统计特征
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

# 截断后统计
print("\n--- power 截断后统计 ---")
for name, df in [('训练集', df_train_pp), ('测试集', df_test_pp)]:
    p = df['power']
    print(f"\n{name}:")
    print(f"  均值:   {p.mean():.1f}")
    print(f"  中位数: {p.median():.1f}")
    print(f"  标准差: {p.std():.1f}")
    print(f"  最小值: {p.min():.1f}")
    print(f"  最大值: {p.max():.1f}")
    print(f"  NaN数:  {p.isna().sum()}")


# ========== 5. 截断后可视化对比 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_train['power'], bins=200, color='salmon', edgecolor='white', alpha=0.7, label='截断前')
axes[0].hist(df_train_pp['power'], bins=200, color='steelblue', edgecolor='white', alpha=0.7, label='截断后')
axes[0].axvline(x=600, color='red', linestyle='--', linewidth=1.5, label='600')
axes[0].set_title('power 截断前后对比')
axes[0].set_xlabel('power')
axes[0].set_ylabel('频数')
axes[0].legend()

axes[1].boxplot(df_train_pp['power'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7))
axes[1].set_title('power 截断后箱线图')
axes[1].set_ylabel('power')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/power_before_after.png', dpi=150)
plt.close()
print(f"\n图表已保存: {OUTPUT_DIR}/power_before_after.png")


# ========== 6. 最终特征一览 ==========
print("\n" + "=" * 60)
print("6. 最终特征一览")
print("=" * 60)

feature_cols = [c for c in df_train_pp.columns if c not in ('SaleID', 'price')]
print(f"特征数量: {len(feature_cols)}")
print(f"特征列表: {feature_cols}")
print(f"训练集形状: {df_train_pp[feature_cols].shape}")
print(f"测试集形状: {df_test_pp[feature_cols].shape}")

print("\n--- 缺失值检查 ---")
for col in feature_cols:
    n_null = df_train_pp[col].isna().sum()
    if n_null > 0:
        print(f"  {col}: {n_null} ({n_null/len(df_train_pp)*100:.2f}%)")

print("\n特征工程完成!")
