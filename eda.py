"""
二手车价格预测 - 探索性数据分析 (EDA)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

OUTPUT_DIR = 'eda_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 1. 加载数据 ==========
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

df_train = pd.read_csv('used_car_train.csv', sep=' ')
df_testA = pd.read_csv('used_car_testA.csv', sep=' ')
df_testB = pd.read_csv('used_car_testB.csv', sep=' ')

print(f"训练集: {df_train.shape}  (含 price)")
print(f"测试集A: {df_testA.shape}")
print(f"测试集B: {df_testB.shape}")

df = df_train

# ========== 2. 数据概览 ==========
print("\n" + "=" * 60)
print("2. 数据概览")
print("=" * 60)

print("\n--- 前5行 ---")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df.head())

print("\n--- 数据类型 ---")
print(df.dtypes)

print("\n--- 统计描述 ---")
print(df.describe().round(4))

# ========== 3. 缺失值分析 ==========
print("\n" + "=" * 60)
print("3. 缺失值分析")
print("=" * 60)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'缺失数量': missing, '缺失比例(%)': missing_pct})
missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失数量', ascending=False)
print(missing_df if len(missing_df) > 0 else "无 NaN 缺失值")

fig, ax = plt.subplots(figsize=(10, 4))
missing_cols = missing_df.index.tolist()
if missing_cols:
    bars = ax.bar(range(len(missing_cols)), missing_df['缺失比例(%)'], color='salmon')
    ax.set_xticks(range(len(missing_cols)))
    ax.set_xticklabels(missing_cols, rotation=30)
    ax.set_ylabel('缺失比例 (%)')
    ax.set_title('训练集缺失值比例')
    for bar, pct in zip(bars, missing_df['缺失比例(%)']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{pct}%', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_missing_values.png', dpi=150)
    plt.close()
    print(f"图表已保存: {OUTPUT_DIR}/01_missing_values.png")

print("\n--- 特殊占位符检查 ('-') ---")
for col in df.columns:
    if df[col].dtype == object:
        dash_count = (df[col] == '-').sum()
        if dash_count > 0:
            print(f"  {col}: {dash_count} 个 '-' ({dash_count / len(df) * 100:.2f}%)")

# ========== 4. 目标变量 price 分布 ==========
print("\n" + "=" * 60)
print("4. 目标变量 price 分布")
print("=" * 60)

print(df['price'].describe())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(df['price'], bins=100, color='steelblue', edgecolor='white')
axes[0].set_title('Price 分布')
axes[0].set_xlabel('Price')
axes[0].set_ylabel('频数')

axes[1].hist(np.log1p(df['price']), bins=100, color='steelblue', edgecolor='white')
axes[1].set_title('log1p(Price) 分布')
axes[1].set_xlabel('log1p(Price)')
axes[1].set_ylabel('频数')

axes[2].boxplot(df['price'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7))
axes[2].set_title('Price 箱线图')
axes[2].set_ylabel('Price')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_price_distribution.png', dpi=150)
plt.close()
print(f"图表已保存: {OUTPUT_DIR}/02_price_distribution.png")

# ========== 5. 类别特征分析 ==========
print("\n" + "=" * 60)
print("5. 类别特征分析")
print("=" * 60)

cat_cols = ['brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'seller', 'offerType']
cat_cols = [c for c in cat_cols if c in df.columns]

fig, axes = plt.subplots(2, 4, figsize=(24, 10))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    vc = df[col].value_counts()
    axes[i].bar(range(len(vc)), vc.values, color='steelblue')
    axes[i].set_xticks(range(len(vc)))
    axes[i].set_xticklabels(vc.index.astype(str), rotation=45, fontsize=8)
    axes[i].set_title(f'{col} 分布 (n={len(vc)})')
    axes[i].set_ylabel('频数')

for j in range(len(cat_cols), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_categorical_features.png', dpi=150)
plt.close()
print(f"图表已保存: {OUTPUT_DIR}/03_categorical_features.png")

fig, axes = plt.subplots(2, 4, figsize=(24, 10))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    price_by_cat = df.groupby(col)['price'].median().sort_values(ascending=False)
    axes[i].bar(range(len(price_by_cat)), price_by_cat.values, color='coral')
    axes[i].set_xticks(range(len(price_by_cat)))
    axes[i].set_xticklabels(price_by_cat.index.astype(str), rotation=45, fontsize=8)
    axes[i].set_title(f'{col} vs Price (中位数)')
    axes[i].set_ylabel('Price 中位数')

for j in range(len(cat_cols), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_categorical_vs_price.png', dpi=150)
plt.close()
print(f"图表已保存: {OUTPUT_DIR}/04_categorical_vs_price.png")

# ========== 6. 数值特征分析 ==========
print("\n" + "=" * 60)
print("6. 数值特征分析")
print("=" * 60)

num_cols = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4',
            'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']
num_cols = [c for c in num_cols if c in df.columns]

fig, axes = plt.subplots(5, 4, figsize=(24, 20))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[i].set_title(f'{col} 分布')
    axes[i].set_ylabel('频数')

for j in range(len(num_cols), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_numeric_distributions.png', dpi=150)
plt.close()
print(f"图表已保存: {OUTPUT_DIR}/05_numeric_distributions.png")

fig, axes = plt.subplots(5, 4, figsize=(24, 20))
axes = axes.flatten()

sample = df.sample(n=min(5000, len(df)), random_state=42)

for i, col in enumerate(num_cols):
    axes[i].scatter(sample[col], sample['price'], alpha=0.3, s=5, color='steelblue')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('price')
    axes[i].set_title(f'{col} vs Price')

for j in range(len(num_cols), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_numeric_vs_price.png', dpi=150)
plt.close()
print(f"图表已保存: {OUTPUT_DIR}/06_numeric_vs_price.png")

# ========== 7. 相关性分析 ==========
print("\n" + "=" * 60)
print("7. 相关性分析")
print("=" * 60)

corr_cols = ['price'] + num_cols
corr_matrix = df[corr_cols].corr()

price_corr = corr_matrix['price'].drop('price').sort_values(ascending=False)
print("\n与 price 的相关系数 (降序):")
print(price_corr.round(4))

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={'size': 7})
ax.set_title('特征相关性热力图')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_correlation_heatmap.png', dpi=150)
plt.close()
print(f"图表已保存: {OUTPUT_DIR}/07_correlation_heatmap.png")

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['coral' if x > 0 else 'steelblue' for x in price_corr.values]
ax.barh(range(len(price_corr)), price_corr.values, color=colors)
ax.set_yticks(range(len(price_corr)))
ax.set_yticklabels(price_corr.index)
ax.set_xlabel('与 Price 的相关系数')
ax.set_title('各特征与 Price 的相关系数')
ax.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_price_correlation_bar.png', dpi=150)
plt.close()
print(f"图表已保存: {OUTPUT_DIR}/08_price_correlation_bar.png")

# ========== 8. 异常值检测 ==========
print("\n" + "=" * 60)
print("8. 异常值检测")
print("=" * 60)

boxplot_cols = ['power', 'kilometer', 'price']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, col in enumerate(boxplot_cols):
    axes[i].boxplot(df[col], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7))
    axes[i].set_title(f'{col} 箱线图')
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_outlier_boxplots.png', dpi=150)
plt.close()
print(f"图表已保存: {OUTPUT_DIR}/09_outlier_boxplots.png")

power_q99 = df['power'].quantile(0.99)
power_q01 = df['power'].quantile(0.01)
print(f"\npower 1%分位: {power_q01}, 99%分位: {power_q99}")
print(f"power > 600 的记录数: {(df['power'] > 600).sum()}")
print(f"power = 0 的记录数: {(df['power'] == 0).sum()}")

# ========== 完成 ==========
print("\n" + "=" * 60)
print("EDA 完成！所有图表已保存到 eda_output/ 目录")
print("=" * 60)
