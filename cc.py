import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ==================== 1. 生成符合Fact 3d逻辑的模拟数据 ====================
np.random.seed(42)
n = 2000  # 总共2000份政策文件

# 城市列表及它们的首次政策年份
city_first_year = {
    '北京': 2009, '上海': 2011, '深圳': 2010,
    '合肥': 2012, '武汉': 2014, '杭州': 2013,
    '广州': 2010, '长沙': 2015, '成都': 2016, '南京': 2012
}

# 随机为每份政策分配城市和年份
cities = list(city_first_year.keys())
df = pd.DataFrame({
    'city': np.random.choice(cities, n),
    'year': np.random.randint(2009, 2023, n)
})

# 计算本地年龄
df['first_year'] = df['city'].map(city_first_year)
df['age'] = df['year'] - df['first_year']
df = df[df['age'] >= 0]  # 只保留首次政策之后的

# ==================== 2. 按Fact 3d的规律生成工具使用概率 ====================
# 核心假设（与论文一致）：
# - 直接补贴：年龄越大，越不可能用（负相关）
# - 研发补贴：年龄越大，越可能用（正相关）
# - 人才政策：年龄越大，越可能用（正相关，但略弱于研发）
# - 土地优惠：年龄越大，越不可能用（负相关）

# 添加随机噪声
noise_subsidy = np.random.normal(0, 0.08, len(df))
noise_rd = np.random.normal(0, 0.08, len(df))
noise_talent = np.random.normal(0, 0.08, len(df))
noise_land = np.random.normal(0, 0.08, len(df))

# 生成概率
df['prob_subsidy'] = (0.7 - 0.04 * df['age'] + noise_subsidy).clip(0, 1)
df['prob_rd'] = (0.2 + 0.05 * df['age'] + noise_rd).clip(0, 1)
df['prob_talent'] = (0.15 + 0.03 * df['age'] + noise_talent).clip(0, 1)
df['prob_land'] = (0.6 - 0.03 * df['age'] + noise_land).clip(0, 1)

# 生成二进制结果
df['tool_subsidy'] = np.random.binomial(1, df['prob_subsidy'])
df['tool_rd'] = np.random.binomial(1, df['prob_rd'])
df['tool_talent'] = np.random.binomial(1, df['prob_talent'])
df['tool_land'] = np.random.binomial(1, df['prob_land'])

print("数据预览：")
print(df[['city', 'year', 'age', 'tool_subsidy', 'tool_rd', 'tool_talent', 'tool_land']].head(10))
# ==================== 3. 回归分析函数 ====================
def run_tool_regression(df, tool_col):
    """
    对特定工具变量跑双向固定效应回归
    """
    y = df[tool_col].values
    X = df[['age']].values

    # 手动加入城市和年份虚拟变量（固定效应）
    city_dummies = pd.get_dummies(df['city'], prefix='city', drop_first=True).astype(float)
    year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True).astype(float)

    X_full = np.column_stack([X, city_dummies.values, year_dummies.values])
    X_full = sm.add_constant(X_full)

    model = sm.OLS(y, X_full).fit()

    # 提取age的系数和标准误
    age_coef = model.params[1]  # 第1个是const，第2个是age
    age_se = model.bse[1]
    age_pval = model.pvalues[1]

    return age_coef, age_se, age_pval, model

# ==================== 4. 跑所有工具的回归 ====================
tools = ['tool_subsidy', 'tool_rd', 'tool_talent', 'tool_land']
tool_labels = {
    'tool_subsidy': '直接财政补贴',
    'tool_rd': '研发与科技应用',
    'tool_talent': '人才引进政策',
    'tool_land': '土地优惠'
}

results = {}
print("\n========== Fact 3d实证回归结果 ==========")
for tool in tools:
    coef, se, pval, model = run_tool_regression(df, tool)
    results[tool] = {'coef': coef, 'se': se, 'pval': pval}
    direction = "促升级（正向）" if coef > 0 else "促进入（负向）"
    print(f"{tool_labels[tool]}: β = {coef:.4f} ({se:.4f}), p = {pval:.4f} → {direction}")

# ==================== 5. 可视化 ====================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, tool in enumerate(tools):
    ax = axes.flatten()[i]

    # 按年龄分组，计算实际使用比例
    age_bins = pd.cut(df['age'], bins=range(0, 16, 2))
    actual_prob = df.groupby(age_bins)[tool].mean()

    # 预测概率
    ages = np.arange(0, 16)
    predicted_prob = 0
    # 简化预测：用回归系数
    coef = results[tool]['coef']
    intercept = df[tool].mean() - coef * df['age'].mean()
    predicted_prob = intercept + coef * ages

    ax.plot(ages, predicted_prob, 'r-', linewidth=2, label='回归拟合')
    actual_prob.plot(ax=ax, kind='bar', alpha=0.4, label='实际值')
    ax.set_xlabel('产业本地年龄')
    ax.set_ylabel('工具使用概率')
    ax.set_title(f'{tool_labels[tool]} (β={coef:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()