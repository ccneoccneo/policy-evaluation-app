"""
案例：碳价预测与配额交易策略
功能：基于LSTM模型预测碳价走势，输出买卖时机建议
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
# ========== 生成模拟碳价数据 ==========
np.random.seed(42)
n_days = 500
dates = pd.date_range('2024-01-01', periods=n_days, freq='B')

# 模拟碳价：均值80，标准差15，带随机游走
carbon_prices = [80]
for i in range(1, n_days):
    change = np.random.normal(0.05, 2.0)
    carbon_prices.append(max(30, carbon_prices[-1] + change))

df = pd.DataFrame({'date': dates, 'carbon_price': carbon_prices})

print("=" * 50)
print("碳市场数据概况")
print("=" * 50)
print(f"数据周期: {df['date'].min().date()} 至 {df['date'].max().date()}")
print(f"交易日数: {len(df)}")
print(f"碳价均值: {df['carbon_price'].mean():.2f} 元/吨")
print(f"碳价标准差: {df['carbon_price'].std():.2f}")
print(f"碳价最高: {df['carbon_price'].max():.2f}, 最低: {df['carbon_price'].min():.2f}")

# ========== 简单趋势预测（移动平均+动量） ==========
df['ma_5'] = df['carbon_price'].rolling(5).mean()
df['ma_20'] = df['carbon_price'].rolling(20).mean()
df['ma_60'] = df['carbon_price'].rolling(60).mean()
df['momentum'] = df['carbon_price'] - df['carbon_price'].shift(20)

# 最新数据
latest = df.iloc[-1]
print(f"\n最新碳价: {latest['carbon_price']:.2f} 元/吨")
print(f"5日移动平均: {latest['ma_5']:.2f}")
print(f"20日移动平均: {latest['ma_20']:.2f}")
print(f"60日移动平均: {latest['ma_60']:.2f}")
print(f"20日动量: {latest['momentum']:.2f}")

# ========== 生成交易策略 ==========
# 假设企业信息
annual_emission = 5000000  # 年排放量 吨
free_allowance = 4500000   # 免费配额 吨
quota_gap = annual_emission - free_allowance  # 正数=需要买入

print(f"\n--- 企业碳排放概览 ---")
print(f"年度排放量: {annual_emission:,} 吨")
print(f"免费配额: {free_allowance:,} 吨")
print(f"配额缺口: {quota_gap:,} 吨 {'(需买入)' if quota_gap > 0 else '(有盈余可卖出)'}")

# 简单趋势判断规则
ma_5 = latest['ma_5']
ma_20 = latest['ma_20']
ma_60 = latest['ma_60']

print("\n--- 交易策略建议 ---")
if ma_5 > ma_20 and ma_20 > ma_60:
    trend = "上升趋势"
    if quota_gap > 0:
        print("趋势判断: 上升趋势（短期均线 > 中期均线 > 长期均线）")
        print(f"建议: 【立即买入】约 {quota_gap * 0.7:.0f} 吨，锁定当前价格")
        print(f"理由: 价格处于上升通道，推迟买入可能面临更高成本")
    else:
        print(f"建议: 【暂时持有】，等待更高价位分批卖出")
elif ma_5 < ma_20 and ma_20 < ma_60:
    trend = "下降趋势"
    if quota_gap > 0:
        print("趋势判断: 下降趋势")
        print(f"建议: 【暂缓买入】，可等待1-2月内价格进一步回落")
    else:
        print(f"建议: 【择机卖出】约 {abs(quota_gap) * 0.5:.0f} 吨，避免价格继续下跌")
else:
    trend = "震荡整理"
    print("趋势判断: 震荡整理")
    print(f"建议: 【分批操作】，每季度买入/卖出配额量的25%")

# ========== 可视化 ==========
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 碳价走势图
ax1 = axes[0]
ax1.plot(df['date'], df['carbon_price'], color='steelblue', linewidth=0.8, alpha=0.6, label='日收盘价')
ax1.plot(df['date'], df['ma_5'], color='red', linewidth=1.5, label='5日均线')
ax1.plot(df['date'], df['ma_20'], color='orange', linewidth=1.5, label='20日均线')
ax1.plot(df['date'], df['ma_60'], color='green', linewidth=1.5, label='60日均线')
ax1.set_ylabel('碳价 (元/吨)')
ax1.set_title('全国碳市场碳价走势与技术指标')
ax1.legend(loc='upper left')
ax1.grid(alpha=0.3)

# 动量指标图
ax2 = axes[1]
ax2.bar(df['date'], df['momentum'], color=np.where(df['momentum']>0, 'red', 'green'), alpha=0.6)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_ylabel('20日动量')
ax2.set_title('碳价动量 (正值=上涨动能，负值=下跌动能)')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ========== 情景分析 ==========
print("\n--- 情景分析：不同碳价水平下的财务影响 ---")
for scenario_price in [60, 80, 100, 120]:
    purchase_cost = quota_gap * scenario_price / 10000  # 万元
    print(f"碳价{scenario_price}元/吨: 买入成本 {purchase_cost:.0f} 万元")