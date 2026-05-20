"""
案例：燃气轮机叶片供应链风险诊断
功能：构建供应链知识图谱，计算供应商综合风险得分，识别国产替代路径
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
# ========== 构建供应链知识图谱 ==========
G = nx.DiGraph()

# 添加节点（企业）
nodes = [
    ('SIEMENS', {'name': '西门子能源', 'country': '德国', 'type': 'OEM'}),
    ('GE', {'name': '通用电气', 'country': '美国', 'type': 'OEM'}),
    ('PCC', {'name': '精密铸件公司', 'country': '美国', 'type': '叶片铸造'}),
    ('HOWMET', {'name': '豪梅特', 'country': '美国', 'type': '叶片精加工'}),
    ('CISRI', {'name': '钢研总院', 'country': '中国', 'type': '材料研发'}),
    ('AECC', {'name': '中国航发', 'country': '中国', 'type': '叶片制造'}),
    ('WZ', {'name': '瓦轴集团', 'country': '中国', 'type': '轴承制造'}),
    ('POWER_GROUP', {'name': '某电力央企', 'country': '中国', 'type': '终端用户'}),
]
G.add_nodes_from(nodes)

# 添加边（供应关系）
edges = [
    ('SIEMENS', 'POWER_GROUP', {'product': '燃气轮机整机', 'dependency': 9, 'substitutability': 3}),
    ('GE', 'POWER_GROUP', {'product': '燃气轮机整机', 'dependency': 7, 'substitutability': 4}),
    ('PCC', 'SIEMENS', {'product': '铸造叶片毛坯', 'dependency': 8, 'substitutability': 3}),
    ('HOWMET', 'SIEMENS', {'product': '精加工叶片', 'dependency': 7, 'substitutability': 3}),
    ('HOWMET', 'GE', {'product': '精加工叶片', 'dependency': 7, 'substitutability': 3}),
    ('CISRI', 'AECC', {'product': '高温合金材料', 'dependency': 4, 'substitutability': 7}),
    ('AECC', 'POWER_GROUP', {'product': '国产替代叶片', 'dependency': 2, 'substitutability': 9}),
    ('WZ', 'POWER_GROUP', {'product': '国产轴承', 'dependency': 3, 'substitutability': 8}),
]
G.add_edges_from(edges)

# 国家风险系数
country_risk = {'中国': 1, '德国': 4, '美国': 8, '日本': 5, '瑞士': 3}

print("=" * 50)
print("供应链风险诊断报告")
print("=" * 50)
print(f"图谱规模: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条供应关系\n")

# ========== 计算每条供应边的综合风险 ==========
risk_records = []
for u, v, data in G.edges(data=True):
    dep = data['dependency']
    sub = data['substitutability']
    # 综合风险 = 依赖度 × (11 - 可替代性) × 国家风险
    supplier_country = G.nodes[u]['country']
    c_risk = country_risk.get(supplier_country, 5)
    risk_score = dep * (11 - sub) * c_risk
    risk_records.append({
        '供应商': G.nodes[u]['name'],
        '供应商国家': supplier_country,
        '客户': G.nodes[v]['name'],
        '供应产品': data['product'],
        '依赖度': dep,
        '可替代性': sub,
        '国家风险': c_risk,
        '综合风险得分': risk_score
    })

df_risk = pd.DataFrame(risk_records).sort_values('综合风险得分', ascending=False)
print("--- 关键供应风险清单（Top 10） ---")
print(df_risk[['供应商', '供应产品', '依赖度', '可替代性', '国家风险', '综合风险得分']].to_string(index=False))

# ========== 风险路径分析 ==========
print("\n--- 关键路径分析 ---")
# 找出风险最大的路径（路径上所有边风险得分之和最大）
max_risk = 0
max_path = None
for u, v, data in G.edges(data=True):
    risk = data['dependency'] * (11 - data['substitutability']) * country_risk.get(G.nodes[u]['country'], 5)
    if risk > max_risk:
        max_risk = risk
        max_path = (G.nodes[u]['name'], G.nodes[v]['name'], data['product'])

print(f"最高风险供应关系: {max_path[0]} → {max_path[1]} ({max_path[2]})，风险得分 {max_risk}")

# ========== 国产替代建议 ==========
print("\n--- 国产替代建议 ---")
chinese_suppliers = [n for n in G.nodes() if G.nodes[n]['country'] == '中国' and G.out_degree(n) > 0]
for cs in chinese_suppliers:
    products = [G[cs][v]['product'] for v in G.successors(cs)]
    print(f"国内供应商: {G.nodes[cs]['name']}，可供应: {', '.join(products)}")

# ========== 可视化 ==========
fig, ax = plt.subplots(figsize=(14, 10))
pos = nx.spring_layout(G, k=3, seed=42)

# 节点颜色按国家
color_map = {'中国': '#2E7D32', '德国': '#1565C0', '美国': '#C62828'}
node_colors = [color_map.get(G.nodes[n]['country'], '#9E9E9E') for n in G.nodes()]
node_sizes = [G.out_degree(n) * 600 + 800 for n in G.nodes()]

# 边宽按依赖度，边色按风险
edge_widths = []
edge_colors = []
for u, v in G.edges():
    dep = G[u][v]['dependency']
    sub = G[u][v]['substitutability']
    c_risk = country_risk.get(G.nodes[u]['country'], 5)
    risk = dep * (11 - sub) * c_risk
    edge_widths.append(dep * 0.8)
    if risk > 200:
        edge_colors.append('#C62828')
    elif risk > 100:
        edge_colors.append('#F9A825')
    else:
        edge_colors.append('#4CAF50')

nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.6, arrowsize=20)
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, edgecolors='white', linewidths=2)
labels = {n: G.nodes[n]['name'] for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=9, font_family='sans-serif')

# 图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E7D32', label='中国 (低风险)'),
    Patch(facecolor='#1565C0', label='德国 (中风险)'),
    Patch(facecolor='#C62828', label='美国 (高风险)'),
    Patch(facecolor='#C62828', alpha=0.6, label='高风险边 (>200)'),
    Patch(facecolor='#F9A825', alpha=0.6, label='中风险边 (100-200)'),
    Patch(facecolor='#4CAF50', alpha=0.6, label='低风险边 (<100)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
ax.set_title('燃气轮机叶片供应链风险地图\n(节点大小=供应规模，边宽=依赖度，边色=风险等级)', fontsize=13)
ax.axis('off')
plt.tight_layout()
plt.show()