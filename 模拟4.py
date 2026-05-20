"""
案例：风光储一体化基地投资风险模拟
功能：通过2000次蒙特卡洛模拟，输出投资回收期的概率分布和敏感性分析
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
# 设置随机种子，保证结果可复现
np.random.seed(42)

# 项目确定性参数
params = {
    'total_investment': 120,  # 总投资，亿元
    'wind_capacity': 2000,  # 风电装机，MW
    'solar_capacity': 1000,  # 光伏装机，MW
    'wind_hours': 2200,  # 风电利用小时
    'solar_hours': 1500,  # 光伏利用小时
    'om_cost_ratio': 0.03,  # 运维成本占投资比例
    'corp_tax_rate': 0.25,  # 所得税率
    'discount_rate': 0.08,  # 折现率
}


def calculate_payback(elec_price, carbon_price):
    """计算单次模拟的投资回收期"""
    # 年度发电收入
    wind_gen = params['wind_capacity'] * params['wind_hours']  # MWh
    solar_gen = params['solar_capacity'] * params['solar_hours']
    total_gen = wind_gen + solar_gen

    revenue = total_gen * elec_price / 10000  # 亿元（电价元/度）
    carbon_rev = total_gen * 0.8 * carbon_price / 1e8  # 亿元（碳价元/吨）

    # 年度成本与税后现金流
    om_cost = params['total_investment'] * params['om_cost_ratio']
    annual_cf = (revenue + carbon_rev - om_cost) * (1 - params['corp_tax_rate'])

    if annual_cf <= 0:
        return 999  # 无法回收
    return params['total_investment'] / annual_cf


# 蒙特卡洛模拟 2000次
n_sim = 2000
payback_results = []
electricity_prices = []
carbon_prices = []

for _ in range(n_sim):
    # 随机抽取不确定参数（正态分布）
    elec = np.random.normal(0.28, 0.08)
    elec = max(0.15, elec)  # 设置下限
    carbon = np.random.normal(80, 30)
    carbon = max(30, carbon)

    payback = calculate_payback(elec, carbon)
    payback_results.append(payback)
    electricity_prices.append(elec)
    carbon_prices.append(carbon)

payback_results = np.array(payback_results)

# ========== 结果输出 ==========
print("=" * 50)
print("风光储一体化基地投资风险模拟结果")
print("=" * 50)
print(f"模拟次数: {n_sim}")
print(f"投资回收期均值: {payback_results.mean():.2f} 年")
print(f"投资回收期中位数: {np.median(payback_results):.2f} 年")
print(f"悲观情景 (95%分位): {np.percentile(payback_results, 95):.2f} 年")
print(f"回收期超过15年的概率: {(payback_results > 15).mean() * 100:.1f}%")
print(f"回收期超过20年的概率: {(payback_results > 20).mean() * 100:.1f}%")

# 敏感性分析
base_case = calculate_payback(0.28, 80)
elec_high = calculate_payback(0.36, 80)  # 电价+1σ
elec_low = calculate_payback(0.20, 80)  # 电价-1σ
carbon_high = calculate_payback(0.28, 110)
carbon_low = calculate_payback(0.28, 50)

print("\n--- 敏感性分析（单因素变动对回收期的影响） ---")
print(f"基准情景: {base_case:.2f} 年")
print(f"电价上升至0.36元/度: {elec_high:.2f} 年 (变化 {elec_high - base_case:+.2f})")
print(f"电价下降至0.20元/度: {elec_low:.2f} 年 (变化 {elec_low - base_case:+.2f})")
print(f"碳价上升至110元/吨: {carbon_high:.2f} 年 (变化 {carbon_high - base_case:+.2f})")
print(f"碳价下降至50元/吨: {carbon_low:.2f} 年 (变化 {carbon_low - base_case:+.2f})")

# 可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 回收期分布直方图
ax1.hist(payback_results, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax1.axvline(payback_results.mean(), color='red', linestyle='--', label=f'均值 {payback_results.mean():.1f}年')
ax1.axvline(15, color='orange', linestyle=':', label='15年警戒线')
ax1.set_xlabel('投资回收期 (年)')
ax1.set_ylabel('频次')
ax1.set_title('2000次蒙特卡洛模拟：投资回收期分布')
ax1.legend()
ax1.grid(alpha=0.3)

# 敏感性分析（旋风图）
factors = ['上网电价', '碳交易价格']
low_impacts = [elec_low - base_case, carbon_low - base_case]
high_impacts = [elec_high - base_case, carbon_high - base_case]

y_pos = range(len(factors))
ax2.barh(y_pos, [abs(x) for x in low_impacts], left=[min(0, x) for x in low_impacts],
         color='lightblue', label='因素下降')
ax2.barh(y_pos, high_impacts, left=0, color='coral', label='因素上升')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(factors)
ax2.set_xlabel('回收期变化 (年)')
ax2.set_title('关键因素敏感性分析 (相对于基准情景)')
ax2.axvline(0, color='black', linewidth=0.5)
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()