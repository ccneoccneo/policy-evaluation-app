"""
案例：重型机械集团数字化转型成熟度诊断
功能：基于企业文本的LLM模拟评分，生成对标雷达图
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
# ========== 模拟企业文本数据 ==========
company_text = """
我公司积极推进数字化转型，已建成覆盖主要生产车间的MES系统，设备联网率达到65%。
在预测性维护方面，已对20台关键设备部署了振动传感器和AI分析模型，故障预警准确率超85%。
ERP系统实现全覆盖，但数据孤岛问题仍存，计划明年启动数据中台建设。
去年成立了数字化部，现有数字化人才120人，占总员工8%。
"""

# ========== LLM模拟评分函数（实际项目替换为真实API调用） ==========
def simulated_llm_scoring(dimension, text):
    """
    模拟大语言模型对企业文本进行评分
    实际项目中，替换为 prompt + API 调用
    """
    # 预设评分规则（模拟）
    rules = {
        '设备联网率': 2.5,
        '数据治理水平': 1.8,
        'AI应用深度': 2.2,
        '工业软件自主率': 1.5,
        '数字化人才占比': 2.0,
        '网络安全防护': 2.8,
    }
    return rules.get(dimension, 1.5)

# 进行评估
dimensions = ['设备联网率', '数据治理水平', 'AI应用深度',
              '工业软件自主率', '数字化人才占比', '网络安全防护']
weights = [0.20, 0.20, 0.15, 0.15, 0.15, 0.15]

company_scores = []
for dim in dimensions:
    score = simulated_llm_scoring(dim, company_text)
    company_scores.append(score)

# 行业标杆水平
benchmark_scores = [3.2, 2.8, 3.0, 2.5, 2.8, 3.5]

# 计算加权总分
total_score = sum(w * s for w, s in zip(weights, company_scores))
total_benchmark = sum(w * s for w, s in zip(weights, benchmark_scores))

print("=" * 50)
print("数字化转型成熟度诊断报告")
print("=" * 50)
print(f"企业加权总分: {total_score:.2f} / 4.0")
print(f"行业标杆总分: {total_benchmark:.2f} / 4.0")
print(f"与标杆差距: {total_benchmark - total_score:.2f}")

print("\n--- 各维度评分 ---")
for dim, cs, bs in zip(dimensions, company_scores, benchmark_scores):
    gap = bs - cs
    flag = "⚠️ 短板" if gap > 0.8 else "✓ 接近"
    print(f"{dim}: 企业{cs:.1f}分 vs 标杆{bs:.1f}分，差距{gap:.1f} {flag}")

# ========== 雷达图 ==========
angles = np.linspace(0, 2*np.pi, len(dimensions), endpoint=False).tolist()
angles += angles[:1]
company_plot = company_scores + company_scores[:1]
benchmark_plot = benchmark_scores + benchmark_scores[:1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))

# 左图：雷达图
ax1.plot(angles, company_plot, 'o-', linewidth=2, label='当前企业', color='steelblue')
ax1.fill(angles, company_plot, alpha=0.15, color='steelblue')
ax1.plot(angles, benchmark_plot, 'o-', linewidth=2, label='行业标杆', color='coral')
ax1.fill(angles, benchmark_plot, alpha=0.15, color='coral')
ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(dimensions, fontsize=9)
ax1.set_ylim(0, 4)
ax1.set_title('数字化成熟度对标', fontsize=12)
ax1.legend(loc='upper right')

# 右图：柱状图
ax2 = fig.add_subplot(1, 2, 2)
x = np.arange(len(dimensions))
width = 0.35
ax2.bar(x - width/2, company_scores, width, label='当前企业', color='steelblue')
ax2.bar(x + width/2, benchmark_scores, width, label='行业标杆', color='coral')
ax2.set_xticks(x)
ax2.set_xticklabels(dimensions, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('得分 (0-4)')
ax2.set_title('各维度得分对比', fontsize=12)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()