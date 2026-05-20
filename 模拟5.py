"""
案例：智能制造试点政策对企业TFP的影响评估
方法：倾向得分匹配（PSM）+ 双重差分（DID）
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
np.random.seed(42)

# ========== 生成模拟数据 ==========
n_firms = 500
df = pd.DataFrame({
    'firm_id': range(n_firms),
    'size': np.random.normal(10, 2, n_firms),          # 企业规模（ln资产）
    'age': np.random.randint(2, 30, n_firms),          # 企业年龄
    'rd_intensity': np.random.beta(2, 5, n_firms)*5,   # 研发强度
    'leverage': np.random.normal(0.5, 0.15, n_firms),  # 资产负债率
})

# 企业入选试点概率（取决于协变量）
df['propensity'] = 1 / (1 + np.exp(-(-2 + 0.3*df['size'] + 0.8*df['rd_intensity'] - 1.5*df['leverage'] + 0.02*df['age'])))
df['treated'] = np.random.binomial(1, df['propensity'])

# 生成TFP（政策效果：处理组TFP额外提升0.15）
df['tfp_pre'] = 2.0 + 0.2*df['size'] + 0.5*df['rd_intensity'] - 0.3*df['leverage'] + np.random.normal(0, 0.1, n_firms)
df['tfp_post'] = df['tfp_pre'] + 0.1 + 0.15*df['treated'] + np.random.normal(0, 0.08, n_firms)

# ========== 第一步：倾向得分匹配 ==========
covariates = ['size', 'age', 'rd_intensity', 'leverage']
logit = LogisticRegression()
logit.fit(df[covariates], df['treated'])
df['pscore'] = logit.predict_proba(df[covariates])[:,1]

# 最近邻匹配（1:1）
treated = df[df['treated']==1].copy()
control = df[df['treated']==0].copy()

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[['pscore']])
distances, indices = nn.kneighbors(treated[['pscore']])
matched_control = control.iloc[indices.flatten()].copy()

print("=" * 50)
print("PSM匹配结果")
print("=" * 50)
print(f"处理组样本数: {len(treated)}")
print(f"匹配后对照组样本数: {len(matched_control)}")

# 匹配前后协变量平衡性检验
print("\n--- 协变量平衡性检验 ---")
for col in covariates:
    t_mean = treated[col].mean()
    c_mean_before = control[col].mean()
    c_mean_after = matched_control[col].mean()
    print(f"{col}: 处理组={t_mean:.3f}, 对照组(匹配前)={c_mean_before:.3f}, 对照组(匹配后)={c_mean_after:.3f}")

# ========== 第二步：DID估计 ==========
treated_diff = treated['tfp_post'].mean() - treated['tfp_pre'].mean()
control_diff = matched_control['tfp_post'].mean() - matched_control['tfp_pre'].mean()
did_estimate = treated_diff - control_diff

print("\n" + "=" * 50)
print("DID估计结果")
print("=" * 50)
print(f"处理组TFP变化: {treated_diff:.4f}")
print(f"对照组TFP变化: {control_diff:.4f}")
print(f"DID估计量（政策净效应）: {did_estimate:.4f}")

# ========== 第三步：回归形式DID ==========
matched_data = pd.concat([treated, matched_control])
matched_data['post'] = 1
pre_data = matched_data.copy()
pre_data['post'] = 0
pre_data['tfp_post'] = pre_data['tfp_pre']  # 把tfp_pre复制到tfp_post位置
panel = pd.concat([matched_data, pre_data])

# 生成交互项
panel['did'] = panel['treated'] * panel['post']
X = panel[['treated', 'post', 'did']]
X = sm.add_constant(X)
y = panel['tfp_post']

model = sm.OLS(y, X).fit()
print("\n--- 回归DID结果 ---")
print(model.summary().tables[1])

# 可视化
fig, ax = plt.subplots(figsize=(8, 5))
groups = ['处理组', '对照组(匹配后)']
pre_means = [treated['tfp_pre'].mean(), matched_control['tfp_pre'].mean()]
post_means = [treated['tfp_post'].mean(), matched_control['tfp_post'].mean()]

x = np.arange(len(groups))
width = 0.35
ax.bar(x - width/2, pre_means, width, label='政策前', color='lightblue')
ax.bar(x + width/2, post_means, width, label='政策后', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel('平均TFP')
ax.set_title(f'DID分析：政策净效应 = {did_estimate:.4f}')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()