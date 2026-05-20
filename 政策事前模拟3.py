import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ============================================
# 1. 生成模拟政策数据集
# ============================================
np.random.seed(42)

cities = ['北京','上海','深圳','合肥','武汉','杭州','广州','长沙','成都','南京']
years = list(range(2010, 2022))

data = []
for city in cities:
    first_year = np.random.choice([2010, 2011, 2012, 2013, 2014])
    for year in years:
        if year < first_year:
            continue
        age = year - first_year
        n_policies = np.random.randint(1, 5)
        for _ in range(n_policies):
            data.append({
                'city': city,
                'year': year,
                'first_year': first_year,
                'age': age
            })

df = pd.DataFrame(data)

# 工具定义
tool_params = {
    'tool_subsidy':        {'alpha0': 0.65, 'beta_true': -0.045, 'label': '直接补贴'},
    'tool_tax_incentive':  {'alpha0': 0.40, 'beta_true': -0.025, 'label': '税收优惠'},
    'tool_land':           {'alpha0': 0.50, 'beta_true': -0.035, 'label': '土地优惠'},
    'tool_rd':             {'alpha0': 0.15, 'beta_true': +0.050, 'label': '研发支持'},
    'tool_talent':         {'alpha0': 0.10, 'beta_true': +0.035, 'label': '人才政策'},
    'tool_supply_chain':   {'alpha0': 0.05, 'beta_true': +0.040, 'label': '供应链协同'},
    'tool_market_access':  {'alpha0': 0.30, 'beta_true': +0.020, 'label': '市场准入与标准'},
    'tool_infra':          {'alpha0': 0.25, 'beta_true': +0.010, 'label': '基建投资'},
    'tool_biz_env':        {'alpha0': 0.20, 'beta_true': +0.030, 'label': '营商环境'},
    'tool_promotion':      {'alpha0': 0.10, 'beta_true': +0.025, 'label': '产业推广'},
}

for tool, params in tool_params.items():
    prob = params['alpha0'] + params['beta_true'] * df['age']
    prob += np.random.normal(0, 0.08, len(df))
    prob = np.clip(prob, 0.01, 0.99)
    df[tool] = np.random.binomial(1, prob)

entry_tools = ['tool_subsidy', 'tool_tax_incentive', 'tool_land']
upgrade_tools = ['tool_rd', 'tool_talent', 'tool_supply_chain',
                 'tool_market_access', 'tool_infra', 'tool_biz_env', 'tool_promotion']

# ============================================
# 2. 估计全国平均规律（Fact 3d模型）
# ============================================
def estimate_age_effect(data, tool_col):
    y = data[tool_col].values
    X_age = data[['age']].values
    city_dum = pd.get_dummies(data['city'], prefix='c', drop_first=True).astype(float)
    year_dum = pd.get_dummies(data['year'], prefix='y', drop_first=True).astype(float)
    X = np.column_stack([X_age, city_dum.values, year_dum.values])
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.params[1], model.bse[1], model.pvalues[1]

betas_estimated = {}
for tool in tool_params.keys():
    coef, se, pval = estimate_age_effect(df, tool)
    betas_estimated[tool] = {'coef': coef, 'se': se, 'pval': pval}

# ============================================
# 3. 诊断模块：以“武汉”为对象
# ============================================
city_name = '武汉'
df_city = df[df['city'] == city_name].copy()
year_target = 2021
df_city_year = df_city[df_city['year'] == year_target]
if len(df_city_year) == 0:
    year_target = df_city['year'].max()
    df_city_year = df_city[df_city['year'] == year_target]

actual = df_city_year[list(tool_params.keys())].mean()
age_current = int(df_city_year['age'].iloc[0])

# 计算预测概率
predicted = {}
for tool, params in tool_params.items():
    pred = params['alpha0'] + betas_estimated[tool]['coef'] * age_current
    predicted[tool] = np.clip(pred, 0, 1)

# ============================================
# 4. 五张诊断图
# ============================================
fig = plt.figure(figsize=(18, 22))

# ---- 图1：政策错配诊断（实际 vs 期望） ----
ax1 = fig.add_subplot(3, 2, 1)
tools_list = list(tool_params.keys())
labels = [tool_params[t]['label'] for t in tools_list]
actual_vals = [actual[t] for t in tools_list]
pred_vals = [predicted[t] for t in tools_list]

x = np.arange(len(labels))
width = 0.35
bars1 = ax1.bar(x - width/2, actual_vals, width, label=f'{city_name}实际值', color='#2196F3')
bars2 = ax1.bar(x + width/2, pred_vals, width, label='全国同龄期望值', color='#FF9800')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('工具使用概率')
ax1.set_title(f'决策一：政策工具错配诊断\n({city_name}, 产业年龄{age_current}年)', fontsize=12)
ax1.legend(fontsize=8)
ax1.grid(axis='y', alpha=0.3)

# ---- 图2：补贴退坡与研发替代（交叉曲线） ----
ax2 = fig.add_subplot(3, 2, 2)
ages_range = np.arange(1, 16)
subsidy_pred = np.clip(tool_params['tool_subsidy']['alpha0'] +
                      betas_estimated['tool_subsidy']['coef'] * ages_range, 0, 1)
rd_pred = np.clip(tool_params['tool_rd']['alpha0'] +
                 betas_estimated['tool_rd']['coef'] * ages_range, 0, 1)

# 找交叉点
cross_age = None
for i in range(len(ages_range)-1):
    if subsidy_pred[i] > rd_pred[i] and subsidy_pred[i+1] <= rd_pred[i+1]:
        cross_age = ages_range[i]
        break

ax2.plot(ages_range, subsidy_pred, 'r-', linewidth=2, label='直接补贴 (促进入)')
ax2.plot(ages_range, rd_pred, 'b-', linewidth=2, label='研发支持 (促升级)')
ax2.axvline(age_current, color='gray', linestyle='--', linewidth=1.5,
            label=f'当前年龄 ({age_current}年)')
if cross_age:
    ax2.axvline(cross_age, color='green', linestyle=':', linewidth=1.5,
                label=f'交叉年龄 ({cross_age}年)')
ax2.set_xlabel('产业本地年龄 (年)')
ax2.set_ylabel('工具使用概率')
ax2.set_title('决策二：补贴退出时机与研发替代', fontsize=12)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# ---- 图3：差异化赛道（竞争城市雷达图） ----
ax3 = fig.add_subplot(3, 2, 3, projection='polar')
comp_cities = ['北京', '上海', '深圳']
compare_tools = ['tool_rd', 'tool_talent', 'tool_supply_chain', 'tool_market_access', 'tool_promotion']
radar_labels = [tool_params[t]['label'] for t in compare_tools]
angles = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False).tolist()
angles += angles[:1]

df_comp = df[df['city'].isin([city_name] + comp_cities)]
city_tool_avg = df_comp.groupby('city')[compare_tools].mean()
colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00']

for idx, city in enumerate([city_name] + comp_cities):
    values = city_tool_avg.loc[city].values.tolist()
    values += values[:1]
    ax3.plot(angles, values, 'o-', linewidth=2, label=city, color=colors[idx])
    ax3.fill(angles, values, alpha=0.1, color=colors[idx])

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(radar_labels, fontsize=8)
ax3.set_title('决策三：差异化赛道分析\n(促升级工具竞争格局)', fontsize=12)
ax3.legend(loc='upper right', fontsize=8)

# ---- 图4：补贴退出预警（趋势+置信区间） ----
ax4 = fig.add_subplot(3, 2, 4)
ages_future = np.arange(1, 16)
subsidy_pred_full = np.clip(tool_params['tool_subsidy']['alpha0'] +
                           betas_estimated['tool_subsidy']['coef'] * ages_future, 0, 1)

# 置信区间（基于β的标准误简化计算）
se_beta = betas_estimated['tool_subsidy']['se']
subsidy_upper = np.clip(tool_params['tool_subsidy']['alpha0'] +
                       (betas_estimated['tool_subsidy']['coef'] + 1.96*se_beta) * ages_future, 0, 1)
subsidy_lower = np.clip(tool_params['tool_subsidy']['alpha0'] +
                       (betas_estimated['tool_subsidy']['coef'] - 1.96*se_beta) * ages_future, 0, 1)

ax4.plot(ages_future, subsidy_pred_full, 'r-', linewidth=2, label='直接补贴预测值')
ax4.fill_between(ages_future, subsidy_lower, subsidy_upper, color='red', alpha=0.2, label='95%置信区间')
ax4.axvline(age_current, color='gray', linestyle='--', linewidth=1.5, label=f'当前年龄 ({age_current}年)')

# 标注未来3年的具体值
for fa in [age_current+1, age_current+2, age_current+3]:
    prob = np.clip(tool_params['tool_subsidy']['alpha0'] +
                  betas_estimated['tool_subsidy']['coef'] * fa, 0, 1)
    ax4.annotate(f'{prob:.2f}', (fa, prob), textcoords="offset points", xytext=(0,10),
                ha='center', fontsize=9, color='darkred')

ax4.set_xlabel('产业本地年龄 (年)')
ax4.set_ylabel('补贴工具使用概率')
ax4.set_title('决策四：企业补贴退出预警', fontsize=12)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# ---- 图5：营商环境优化（工具结构演变） ----
ax5 = fig.add_subplot(3, 2, 5)
# 计算“促进入”和“促升级”综合得分随年龄的变化
entry_pred_curve = np.zeros(len(ages_range))
upgrade_pred_curve = np.zeros(len(ages_range))
for i, age in enumerate(ages_range):
    e_score = 0
    u_score = 0
    for t in entry_tools:
        prob = np.clip(tool_params[t]['alpha0'] + betas_estimated[t]['coef'] * age, 0, 1)
        e_score += prob
    for t in upgrade_tools:
        prob = np.clip(tool_params[t]['alpha0'] + betas_estimated[t]['coef'] * age, 0, 1)
        u_score += prob
    entry_pred_curve[i] = e_score / len(entry_tools)
    upgrade_pred_curve[i] = u_score / len(upgrade_tools)

ax5.plot(ages_range, entry_pred_curve, 'r-', linewidth=2, label='促进入工具 (补贴+土地+税收)')
ax5.plot(ages_range, upgrade_pred_curve, 'b-', linewidth=2, label='促升级工具 (研发+人才+供应链...)')
ax5.axvline(age_current, color='gray', linestyle='--', linewidth=1.5, label=f'当前年龄 ({age_current}年)')

# 找主导权转换点
for i in range(len(ages_range)):
    if upgrade_pred_curve[i] > entry_pred_curve[i]:
        ax5.axvline(ages_range[i], color='green', linestyle=':', linewidth=1.5,
                    label=f'升级主导起点 ({ages_range[i]}年)')
        break

ax5.set_xlabel('产业本地年龄 (年)')
ax5.set_ylabel('工具平均使用概率')
ax5.set_title('决策五：从"给钱"到"给服务"\n(工具结构随产业年龄演变)', fontsize=12)
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

# ---- 图6：Summary表 ----
# ---- 图6：结论总表（使用 Matplotlib 原生表格，完全避免字符画乱码） ----
ax6 = fig.add_subplot(3, 2, 6)
ax6.axis('off')

# 重新计算所有判断条件需要的变量（确保在作用域内）
# 决策一：错配判断
mismatch_flag = any(abs(actual[t] - predicted[t]) > 0.15 for t in tools_list)

# 决策二：补贴退坡判断
cross_judge = (cross_age is not None) and (age_current >= cross_age)

# 决策三：差异化赛道判断
# 重新计算与竞争对手的差距
other_avg = city_tool_avg.drop(city_name).mean()
my_avg = city_tool_avg.loc[city_name]
gap_diff = my_avg - other_avg
weakest_tool = gap_diff.idxmin() if gap_diff.min() < -0.05 else None
diff_flag = (gap_diff.min() < -0.05)

# 决策四：企业预警判断（补贴概率阈值）
subsidy_prob_current = min(subsidy_pred_full[min(age_current, 14)], 1.0)  # 防止索引越界
warning_flag = (subsidy_prob_current < 0.3)

# 决策五：营商环境判断（升级工具得分是否超过进入工具得分）
# 使用当前年龄对应的曲线值（已计算过 entry_pred_curve 和 upgrade_pred_curve）
idx_current = min(age_current, len(ages_range) - 1)
service_flag = (upgrade_pred_curve[idx_current] > entry_pred_curve[idx_current])

# 构建表格内容
table_data = [
    ['诊断项目', '当前状态', '核心建议'],
    ['决策一：政策错配',
     '存在错配' if mismatch_flag else '基本正常',
     '调整工具结构' if mismatch_flag else '维持现状'],
    ['决策二：补贴退坡',
     '已超交叉点' if cross_judge else '尚未到达',
     '补贴转研发' if cross_judge else '暂维持补贴'],
    ['决策三：差异化赛道',
     '存在薄弱环节' if diff_flag else '竞争格局平衡',
     f'聚焦{weakest_tool}领域' if diff_flag else '巩固现有优势'],
    ['决策四：企业预警',
     '补贴将快速退出' if warning_flag else '暂时安全',
     '按无补贴情景测算' if warning_flag else '持续观察'],
    ['决策五：营商环境',
     '应转向服务型' if service_flag else '维持激励型',
     '减补贴，加服务' if service_flag else '维持现有激励'],
]

# 创建表格
table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.22, 0.22, 0.26])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# 表头样式
for j in range(3):
    cell = table[0, j]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(color='white', fontweight='bold')

# 数据行交替颜色
for i in range(1, len(table_data)):
    for j in range(3):
        cell = table[i, j]
        if i % 2 == 0:
            cell.set_facecolor('#D6E4F0')
        else:
            cell.set_facecolor('#FFFFFF')

ax6.set_title(f'{city_name}新能源汽车产业政策 Fact 3d 诊断总表',
              fontsize=12, fontweight='bold', y=1.02)

plt.tight_layout()
plt.show()