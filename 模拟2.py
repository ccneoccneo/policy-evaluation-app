import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. 生成模拟政策数据集（在真实项目中由LLM输出）
# --------------------------------------------
np.random.seed(42)

cities = ['北京','上海','深圳','合肥','武汉','杭州','广州','长沙','成都','南京']
years = list(range(2010, 2022))

data = []
# 每个城市首次出台政策的年份随机分布于2010-2014
for city in cities:
    first_year = np.random.choice([2010, 2011, 2012, 2013, 2014])
    for year in years:
        if year < first_year:
            continue
        age = year - first_year
        n_policies = np.random.randint(1, 5)   # 每年1-4份政策
        for _ in range(n_policies):
            data.append({
                'city': city,
                'year': year,
                'first_year': first_year,
                'age': age
            })

df = pd.DataFrame(data)

# 定义工具及其在“生命周期”中的真实影响系数
# alpha0：年龄为0时的基础概率；beta：年龄效应（负值=随年龄下降，正值=随年龄上升）
tool_params = {
    'tool_subsidy':        {'alpha0': 0.65, 'beta_true': -0.045},   # 直接补贴
    'tool_tax_incentive':  {'alpha0': 0.40, 'beta_true': -0.025},   # 税收优惠
    'tool_land':           {'alpha0': 0.50, 'beta_true': -0.035},   # 土地优惠
    'tool_rd':             {'alpha0': 0.15, 'beta_true': +0.050},   # 研发支持
    'tool_talent':         {'alpha0': 0.10, 'beta_true': +0.035},   # 人才政策
    'tool_supply_chain':   {'alpha0': 0.05, 'beta_true': +0.040},   # 供应链协同
    'tool_market_access':  {'alpha0': 0.30, 'beta_true': +0.020},   # 市场准入与标准
    'tool_infra':          {'alpha0': 0.25, 'beta_true': +0.010},   # 基建投资
    'tool_biz_env':        {'alpha0': 0.20, 'beta_true': +0.030},   # 营商环境
    'tool_promotion':      {'alpha0': 0.10, 'beta_true': +0.025},   # 产业推广/出海
}

# 生成每个工具的使用情况（0/1）
for tool, params in tool_params.items():
    prob = params['alpha0'] + params['beta_true'] * df['age']
    prob += np.random.normal(0, 0.08, len(df))    # 添加随机扰动
    prob = np.clip(prob, 0.01, 0.99)
    df[tool] = np.random.binomial(1, prob)

# 工具分类
entry_tools = ['tool_subsidy', 'tool_tax_incentive', 'tool_land']
upgrade_tools = ['tool_rd', 'tool_talent', 'tool_supply_chain',
                 'tool_market_access', 'tool_infra', 'tool_biz_env', 'tool_promotion']

# --------------------------------------------
# 2. 估计全国平均规律（Fact 3d的双向固定效应模型）
# --------------------------------------------
def estimate_age_effect(data, tool_col):
    """对指定工具估计年龄的边际效应（β）"""
    y = data[tool_col].values
    X_age = data[['age']].values

    # 城市和年份虚拟变量（固定效应）
    city_dum = pd.get_dummies(data['city'], prefix='c', drop_first=True).astype(float)
    year_dum = pd.get_dummies(data['year'], prefix='y', drop_first=True).astype(float)
    X = np.column_stack([X_age, city_dum.values, year_dum.values])
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    return model.params[1], model.bse[1], model.pvalues[1]

# 估计所有工具的β系数
betas_estimated = {}
for tool in tool_params.keys():
    coef, se, pval = estimate_age_effect(df, tool)
    betas_estimated[tool] = {'coef': coef, 'se': se, 'pval': pval}

print("=" * 60)
print("Fact 3d 全国平均规律（β系数）")
print("=" * 60)
for tool, res in betas_estimated.items():
    direction = "促升级" if res['coef'] > 0 else "促进入"
    print(f"{tool:25s}: β = {res['coef']:+7.4f} (se={res['se']:.4f}), p={res['pval']:.4f} → {direction}")

# --------------------------------------------
# 3. 诊断模块：以“武汉”为例
# --------------------------------------------
city_name = '武汉'
df_city = df[df['city'] == city_name].copy()

# 查询2021年的数据
year_target = 2021
df_city_year = df_city[df_city['year'] == year_target]
if len(df_city_year) == 0:
    # 若无准确年份，取可用最近年份
    year_target = df_city['year'].max()
    df_city_year = df_city[df_city['year'] == year_target]

actual = df_city_year[list(tool_params.keys())].mean()    # 实际使用率
age_current = int(df_city_year['age'].iloc[0])

print("\n" + "=" * 60)
print(f"诊断对象：{city_name}，目标年份：{year_target}，产业本地年龄：{age_current} 年")
print("=" * 60)

# ---- 决策一：政策错配诊断 ----
print("\n【决策一】政策工具错配诊断")
print("-" * 40)
for tool, params in tool_params.items():
    pred_prob = params['alpha0'] + betas_estimated[tool]['coef'] * age_current
    pred_prob = np.clip(pred_prob, 0, 1)
    gap = actual[tool] - pred_prob

    flag = ""
    if gap > 0.15:
        flag = "⚠️ 过度使用"
        if betas_estimated[tool]['coef'] < 0:
            flag += " (建议削减/退出)"
    elif gap < -0.15:
        flag = "⚠️ 使用不足"
        if betas_estimated[tool]['coef'] > 0:
            flag += " (建议加强)"
    else:
        flag = "✓ 正常"

    print(f"{tool:25s} | 实际: {actual[tool]:.2f} | 期望: {pred_prob:.2f} | 差距: {gap:+.2f} | {flag}")

# ---- 决策二：补贴退坡与研发替代 ----
print("\n【决策二】补贴退出时机与研发替代建议")
print("-" * 40)
ages_range = np.arange(1, 16)
subsidy_pred = np.clip(tool_params['tool_subsidy']['alpha0'] +
                      betas_estimated['tool_subsidy']['coef'] * ages_range, 0, 1)
rd_pred = np.clip(tool_params['tool_rd']['alpha0'] +
                 betas_estimated['tool_rd']['coef'] * ages_range, 0, 1)

# 寻找交叉年龄
cross_age = None
for i in range(len(ages_range)-1):
    if subsidy_pred[i] > rd_pred[i] and subsidy_pred[i+1] <= rd_pred[i+1]:
        cross_age = ages_range[i]
        break

print(f"研发工具使用概率超越补贴工具的年龄节点约为: {cross_age} 年")
print(f"{city_name}当前新能源产业年龄: {age_current} 年")
if cross_age and age_current >= cross_age:
    print("▶ 结论：已落入研发主导区间，建议加速将建厂补贴预算转投公共研发平台。")
else:
    print(f"▶ 结论：可暂时维持现有补贴力度，但应在第{cross_age}年前逐步启动转型。")

# ---- 决策三：差异化赛道分析 ----
print("\n【决策三】差异化赛道选择（与竞争城市对比）")
print("-" * 40)
comp_cities = ['北京','上海','深圳']
df_comp = df[df['city'].isin([city_name] + comp_cities)]
city_tool_avg = df_comp.groupby('city')[['tool_rd', 'tool_supply_chain', 'tool_market_access']].mean()

print("各城市促升级工具平均使用率：")
print(city_tool_avg.round(3))

my_avg = city_tool_avg.loc[city_name]
other_avg = city_tool_avg.drop(city_name).mean()
gap_diff = my_avg - other_avg
print(f"\n{city_name}相比竞争对手的差异（正值=相对领先，负值=相对落后）：")
print(gap_diff)

weakest_tool = gap_diff.idxmin()
if gap_diff[weakest_tool] < -0.05:
    print(f"▶ 建议：在“{weakest_tool}”领域明显落后，可结合本地优势（如商用车底盘、电池回收）打造差异化赛道。")

# ---- 决策四：企业补贴退出预警 ----
print("\n【决策四】企业投资补贴退出预警")
print("-" * 40)
future_ages = [age_current + 1, age_current + 2, age_current + 3]
for fa in future_ages:
    prob = np.clip(tool_params['tool_subsidy']['alpha0'] +
                  betas_estimated['tool_subsidy']['coef'] * fa, 0, 1)
    print(f"  未来第{fa-age_current}年（产业年龄 {fa}）：直接补贴出现概率预计为 {prob:.2f}")

print("▶ 建议：企业应按照‘补贴将在2-3年内大幅退出’的假设进行投资回报测算。")

# ---- 决策五：营商环境优化 ----
print("\n【决策五】从“给钱”到“给服务”——营商环境诊断")
print("-" * 40)
# 简化判断：当升级工具的平均使用率开始稳定超过进入工具时
entry_pred_curve = np.clip(tool_params['tool_subsidy']['alpha0'] +
                          betas_estimated['tool_subsidy']['coef'] * ages_range, 0, 1)
upgrade_pred_curve = np.clip(tool_params['tool_rd']['alpha0'] +
                            betas_estimated['tool_rd']['coef'] * ages_range, 0, 1)

dominant_age = None
for i in range(len(ages_range)):
    if upgrade_pred_curve[i] > entry_pred_curve[i]:
        dominant_age = ages_range[i]
        break

print(f"促升级工具（以研发为代表）超越促进入工具（以补贴为代表）的年龄阈值: {dominant_age} 年")
if dominant_age and age_current >= dominant_age:
    print(f"▶ 建议：{city_name}已进入‘服务主导’阶段，应大幅减少直接补贴，重点建设公共测试平台、知识产权保护、组织出海参展等。")
else:
    print(f"▶ 建议：当前仍需维持一定的直接激励，但应开始布局服务型政策工具。")

# ---- 可视化：核心工具年龄效应 ----
plt.figure(figsize=(10, 6))
plt.plot(ages_range, subsidy_pred, 'r-', linewidth=2, label='直接补贴 (促进入)')
plt.plot(ages_range, rd_pred, 'b-', linewidth=2, label='研发与科技应用 (促升级)')
plt.axvline(age_current, color='gray', linestyle='--', label=f'{city_name}当前年龄({age_current}年)')
if cross_age:
    plt.axvline(cross_age, color='green', linestyle=':', label=f'交叉年龄({cross_age}年)')
plt.xlabel('产业本地年龄', fontsize=12)
plt.ylabel('工具使用概率', fontsize=12)
plt.title('新能源汽车产业政策工具生命周期（Fact 3d实证规律）', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()