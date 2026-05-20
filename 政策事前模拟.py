"""
新能源汽车产业政策事前模拟系统
基于系统动力学方法，模拟补贴结构调整对产业演化的长期影响
包含确定性模拟 + 蒙特卡洛敏感性分析
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from typing import Dict, Tuple
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


# ==================== 系统微分方程定义 ====================
def new_energy_vehicle_system(state, t, params):
    """
    新能源汽车产业政策模拟系统
    状态变量: [Firms, Tech, Chargers]
    """
    F, T, C = state

    # ---- 政策参数 ----
    total_budget = params['total_budget']
    base_subsidy_ratio = params['base_subsidy_ratio']
    adjust_start = params['adjust_start']
    adjust_end = params['adjust_end']
    target_subsidy_ratio = params['target_subsidy_ratio']
    r_d_share = params['r_d_share']
    charger_share = params['charger_share']

    # 政策比例随时间变化（线性过渡）
    if t < adjust_start:
        subsidy_ratio = base_subsidy_ratio
    elif t > adjust_end:
        subsidy_ratio = target_subsidy_ratio
    else:
        frac = (t - adjust_start) / (adjust_end - adjust_start)
        subsidy_ratio = base_subsidy_ratio + frac * (target_subsidy_ratio - base_subsidy_ratio)

    # 各项投入分配
    production_subsidy = total_budget * subsidy_ratio
    r_d_budget = total_budget * (1 - subsidy_ratio) * r_d_share
    charger_invest = total_budget * (1 - subsidy_ratio) * charger_share

    # ---- 补贴吸引效应（边际递减） ----
    subsidy_effect = production_subsidy / total_budget
    subsidy_attraction = params['w_subsidy'] * np.log(1 + subsidy_effect * 10) / np.log(11)

    # ---- 技术生态效应 ----
    avg_tech = T / F if F > 0 else 0
    tech_effect = min(1.0, avg_tech / params['target_tech_level'])
    tech_attraction = params['w_tech'] * tech_effect

    # ---- 充电桩效应 ----
    charger_effect = min(1.0, C / params['target_chargers'])
    charger_attraction = params['w_charger'] * charger_effect

    # ---- 产业吸引力综合 ----
    attraction = 1.0 + subsidy_attraction + tech_attraction + charger_attraction

    # ---- 企业进入与退出 ----
    entry_rate = params['base_entry'] * attraction
    natural_exit = params['natural_exit_rate'] * F
    over_capacity = max(0, F - params['carrying_capacity'])
    competitive_exit = params['competitive_exit_coeff'] * over_capacity * F / params['carrying_capacity']
    exit_rate = natural_exit + competitive_exit
    dF_dt = entry_rate - exit_rate

    # ---- 技术知识增长 ----
    firm_rd = params['firm_rd_per_capita'] * F
    public_rd = r_d_budget * params['rd_conversion']
    dT_dt = firm_rd + public_rd

    # ---- 充电桩动态 ----
    charger_build = charger_invest * params['charger_conversion']
    charger_depreciation = params['charger_depreciation_rate'] * C
    dC_dt = charger_build - charger_depreciation

    return [dF_dt, dT_dt, dC_dt]


# ==================== 参数与初始状态 ====================
params_default = {
    'total_budget': 100000,  # 年财政总投入（万元）
    'base_subsidy_ratio': 0.7,  # 基准补贴占比
    'target_subsidy_ratio': 0.4,  # 调整后补贴占比
    'adjust_start': 3.0,  # 调整开始年份
    'adjust_end': 5.0,  # 调整完成年份
    'r_d_share': 0.6,  # 研发占非补贴投入比例
    'charger_share': 0.4,  # 充电桩占非补贴投入比例
    'w_subsidy': 0.6,  # 补贴吸引力权重
    'w_tech': 0.4,  # 技术吸引力权重
    'w_charger': 0.3,  # 充电桩吸引力权重
    'target_tech_level': 50,  # 目标技术水平（专利当量/家）
    'target_chargers': 5000,  # 目标充电桩保有量（个）
    'base_entry': 10,  # 基础进入速率（家/年）
    'natural_exit_rate': 0.05,  # 自然退出率
    'carrying_capacity': 300,  # 环境承载力（家）
    'competitive_exit_coeff': 0.2,  # 竞争挤出系数
    'firm_rd_per_capita': 0.1,  # 企业自发研发贡献（专利当量/家·年）
    'rd_conversion': 0.00005,  # 公共研发转化系数（专利当量/万元）
    'charger_conversion': 0.0002,  # 充电桩建设系数（个/万元）
    'charger_depreciation_rate': 0.03,  # 充电桩折旧率
}

initial_state = [50, 100, 1000]  # [企业数, 技术知识总量, 充电桩总量]


# ==================== 确定性模拟：基准情景 vs 实验情景 ====================
def run_deterministic_simulation():
    # 基准情景：永不调整
    params_base = params_default.copy()
    params_base['adjust_start'] = 100.0
    params_base['adjust_end'] = 101.0

    # 实验情景：第3-5年调整
    params_expt = params_default.copy()

    t = np.linspace(0, 10, 200)  # 0~10年，200个点
    sol_base = odeint(new_energy_vehicle_system, initial_state, t, args=(params_base,))
    sol_expt = odeint(new_energy_vehicle_system, initial_state, t, args=(params_expt,))

    F_b, T_b, C_b = sol_base.T
    F_e, T_e, C_e = sol_expt.T
    avg_tech_b = T_b / F_b
    avg_tech_e = T_e / F_e

    # 可视化对比
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    titles = ['企业总数 (Firms)', '技术知识总量 (Tech)', '平均技术水平 (Avg Tech)', '充电桩总量 (Chargers)']
    data_b = [F_b, T_b, avg_tech_b, C_b]
    data_e = [F_e, T_e, avg_tech_e, C_e]
    ylabels = ['家', '专利当量', '专利当量/家', '个']

    for i, ax in enumerate(axes.flatten()):
        ax.plot(t, data_b[i], 'b-', linewidth=2, label='基准情景 (70%补贴不变)')
        ax.plot(t, data_e[i], 'r--', linewidth=2, label='实验情景 (降至40%)')
        ax.set_xlabel('时间 (年)')
        ax.set_ylabel(ylabels[i])
        ax.set_title(titles[i])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.axvline(5, color='gray', linestyle=':', alpha=0.7)
            ax.text(5.1, ax.get_ylim()[1] * 0.92, '调整完成', fontsize=9)

    plt.tight_layout()
    plt.show()

    # 输出关键节点数值
    print("=== 确定性模拟关键指标 ===")
    for year in [3, 5, 8, 10]:
        idx = np.argmin(np.abs(t - year))
        print(f"第{year}年:")
        print(f"  基准 - 企业数: {F_b[idx]:.1f}, 平均技术: {avg_tech_b[idx]:.2f}, 充电桩: {C_b[idx]:.0f}")
        print(f"  实验 - 企业数: {F_e[idx]:.1f}, 平均技术: {avg_tech_e[idx]:.2f}, 充电桩: {C_e[idx]:.0f}")
        print()


# ==================== 蒙特卡洛敏感性分析 ====================
def monte_carlo_sensitivity(n_simulations=200):
    """
    对关键参数随机采样，比较第10年平均技术水平
    返回实验情景优于基准情景的概率
    """
    np.random.seed(42)
    tech_base_list, tech_expt_list = [], []
    firms_base_list, firms_expt_list = [], []

    for _ in range(n_simulations):
        # 扰动关键参数
        params_mc = params_default.copy()
        params_mc['w_subsidy'] = np.random.uniform(0.4, 0.8)
        params_mc['w_tech'] = np.random.uniform(0.3, 0.6)
        params_mc['w_charger'] = np.random.uniform(0.2, 0.5)
        params_mc['rd_conversion'] = np.random.uniform(0.00003, 0.00008)
        params_mc['carrying_capacity'] = np.random.uniform(250, 400)
        params_mc['firm_rd_per_capita'] = np.random.uniform(0.05, 0.15)
        params_mc['charger_conversion'] = np.random.uniform(0.00015, 0.00025)
        adj_start = np.random.uniform(2.5, 3.5)
        params_mc['adjust_start'] = adj_start
        params_mc['adjust_end'] = adj_start + np.random.uniform(1.5, 2.5)

        # 基准情景（不调整）
        params_mc_base = params_mc.copy()
        params_mc_base['adjust_start'] = 100.0

        t_mc = np.linspace(0, 10, 100)
        sol_base = odeint(new_energy_vehicle_system, initial_state, t_mc, args=(params_mc_base,))
        sol_expt = odeint(new_energy_vehicle_system, initial_state, t_mc, args=(params_mc,))

        tech_base_list.append(sol_base[-1, 1] / sol_base[-1, 0])
        tech_expt_list.append(sol_expt[-1, 1] / sol_expt[-1, 0])
        firms_base_list.append(sol_base[-1, 0])
        firms_expt_list.append(sol_expt[-1, 0])

    # 统计检验
    print("=== 蒙特卡洛敏感性分析 ({}次模拟) ===".format(n_simulations))
    print(f"第10年平均技术水平（专利当量/家）:")
    print(f"  基准情景: 均值 {np.mean(tech_base_list):.2f}, "
          f"95%CI [{np.percentile(tech_base_list, 2.5):.2f}, {np.percentile(tech_base_list, 97.5):.2f}]")
    print(f"  实验情景: 均值 {np.mean(tech_expt_list):.2f}, "
          f"95%CI [{np.percentile(tech_expt_list, 2.5):.2f}, {np.percentile(tech_expt_list, 97.5):.2f}]")

    diff = np.array(tech_expt_list) - np.array(tech_base_list)
    prob_positive = np.mean(diff > 0) * 100
    print(f"\n实验情景平均技术水平高于基准的概率: {prob_positive:.1f}%")

    # 直方图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(tech_base_list, bins=20, alpha=0.6, label='基准情景', color='blue', density=True)
    ax1.hist(tech_expt_list, bins=20, alpha=0.6, label='实验情景', color='red', density=True)
    ax1.axvline(np.mean(tech_base_list), color='blue', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(tech_expt_list), color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('平均技术水平 (专利当量/家)')
    ax1.set_ylabel('概率密度')
    ax1.set_title('第10年技术水平分布')
    ax1.legend()

    ax2.hist(diff, bins=20, color='green', alpha=0.7, density=True)
    ax2.axvline(0, color='black', linestyle='--', linewidth=2)
    ax2.set_xlabel('实验情景 - 基准情景 (差异)')
    ax2.set_title('技术水平差异分布')
    plt.tight_layout()
    plt.show()

    return prob_positive


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    print("===== 产业政策飞行模拟器：新能源汽车 =====")
    print("1. 确定性情景对比模拟")
    run_deterministic_simulation()
    print("\n2. 蒙特卡洛敏感性分析")
    prob = monte_carlo_sensitivity(200)
    print(f"\n结论: 政策转向使技术水平提升的可能性为 {prob:.1f}%")