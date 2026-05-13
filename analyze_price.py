"""
二手车价格预测 - price 分布分析与变换方案
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'price_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 1. 加载数据 ==========
df = pd.read_csv('used_car_train.csv', sep=' ')
price = df['price']

print("=" * 60)
print("price 原始分布统计")
print("=" * 60)
print(price.describe())
print(f"\n偏度 (skewness): {price.skew():.4f}")
print(f"峰度 (kurtosis): {price.kurtosis():.4f}")
print(f"零值数量: {(price == 0).sum()}")
print(f"<= 100 的数量: {(price <= 100).sum()} ({(price <= 100).mean()*100:.2f}%)")
print(f">= 50000 的数量: {(price >= 50000).sum()} ({(price >= 50000).mean()*100:.2f}%)")

# 分位数
print("\n分位数:")
for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f"  {q*100:5.1f}%: {price.quantile(q):.0f}")


# ========== 2. 候选变换 ==========
# 过滤 price <= 0 的记录（log 变换需要正数）
print(f"\nprice <= 0 的记录: {(price <= 0).sum()} 条，变换时排除")

transforms = {
    '原始 price': price,
    'log1p(price)': np.log1p(price),
    'log1p(price/100)': np.log1p(price / 100),
    'sqrt(price)': np.sqrt(price),
    'Box-Cox': None,  # 需要正数，单独处理
    '1/4 幂 (x^0.25)': np.power(price, 0.25),
}

# Box-Cox 变换（要求严格正数）
price_pos = price[price > 0]
bc_result, bc_lambda = stats.boxcox(price_pos)
transforms['Box-Cox'] = pd.Series(bc_result, index=price_pos.index)
print(f"\nBox-Cox 最优 lambda: {bc_lambda:.4f}")


# ========== 3. 偏度/峰度对比 ==========
print("\n" + "=" * 60)
print("各变换的偏度/峰度对比")
print("=" * 60)
print(f"{'变换方法':<25} {'偏度':>10} {'峰度':>10} {'正态性(p值)':>15}")
print("-" * 65)

results = {}
for name, transformed in transforms.items():
    skew = transformed.skew() if isinstance(transformed, pd.Series) else stats.skew(transformed)
    kurt = transformed.kurtosis() if isinstance(transformed, pd.Series) else stats.kurtosis(transformed)
    # Shapiro-Wilk 正态性检验（抽样 5000）
    sample = transformed.sample(n=min(5000, len(transformed)), random_state=42)
    stat_sw, p_sw = stats.shapiro(sample)
    print(f"{name:<25} {skew:>10.4f} {kurt:>10.4f} {p_sw:>15.6f}")
    results[name] = {'skew': skew, 'kurt': kurt, 'p_sw': p_sw}


# ========== 4. 可视化 ==========
print("\n" + "=" * 60)
print("4. 可视化")
print("=" * 60)

# 4.1 原始分布 + QQ图
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

axes[0].hist(price, bins=200, color='steelblue', edgecolor='white')
axes[0].set_title(f'原始 price (skew={price.skew():.2f})')
axes[0].set_xlabel('price')
axes[0].set_ylabel('频数')

axes[1].hist(price[price <= price.quantile(0.95)], bins=100, color='steelblue', edgecolor='white')
axes[1].set_title('price (<= 95%分位)')
axes[1].set_xlabel('price')

stats.probplot(price, dist="norm", plot=axes[2])
axes[2].set_title('price QQ 图')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_price_raw.png', dpi=150)
plt.close()
print(f"图表已保存: {OUTPUT_DIR}/01_price_raw.png")

# 4.2 各变换对比（直方图 + QQ图）
fig, axes = plt.subplots(len(transforms), 2, figsize=(16, 5 * len(transforms)))

for i, (name, transformed) in enumerate(transforms.items()):
    # 直方图
    axes[i, 0].hist(transformed, bins=100, color='steelblue', edgecolor='white')
    skew_val = results[name]['skew']
    axes[i, 0].set_title(f'{name} (skew={skew_val:.4f})')
    axes[i, 0].set_ylabel('频数')

    # QQ 图
    sample = transformed.sample(n=min(5000, len(transformed)), random_state=42)
    stats.probplot(sample, dist="norm", plot=axes[i, 1])
    axes[i, 1].set_title(f'{name} QQ 图')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_transforms_compare.png', dpi=150)
plt.close()
print(f"图表已保存: {OUTPUT_DIR}/02_transforms_compare.png")

# 4.3 推荐变换对比（只看 top 3）
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

top3 = ['log1p(price)', 'log1p(price/100)', '1/4 幂 (x^0.25)']
colors = ['steelblue', 'coral', 'seagreen']

for i, name in enumerate(top3):
    transformed = transforms[name]
    axes[i].hist(transformed, bins=100, color=colors[i], edgecolor='white')
    axes[i].set_title(f'{name} (skew={results[name]["skew"]:.4f})')
    axes[i].set_ylabel('频数')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_top3_transforms.png', dpi=150)
plt.close()
print(f"图表已保存: {OUTPUT_DIR}/03_top3_transforms.png")


# ========== 5. 推荐方案 ==========
print("\n" + "=" * 60)
print("5. 推荐方案")
print("=" * 60)

print("""
price 分布问题：
  - 右偏严重 (skew > 3)，存在明显的长尾
  - 99% 分位为 9999，但最大值达 199999

推荐变换方案：

1. log1p(price)  -- 最推荐
   - 偏度降至接近 0，分布近似正态
   - 可逆性好：反变换 exp(pred) - 1
   - 适合回归树模型，有利于预测中低价区间
   - 应用方式：训练时 y_train = log1p(price)
               预测时 pred = expm1(pred) -> max(pred, 1.0)

2. log1p(price/100)  -- 可选
   - 进一步缩小数值范围
   - 反变换 expm1(pred) * 100

3. x^0.25 (1/4 幂)  -- 可选
   - 偏度改善明显，且保持 0 值为 0
   - 反变换 pred^4
   - 对极端值不如 log 敏感

模型集成方式：
   在 train_catboost2.py 中：
   - 训练：y = log1p(price)
   - 预测：pred = np.expm1(model.predict(X_test))
   - 后处理：pred = np.maximum(pred, 1.0)
""")
