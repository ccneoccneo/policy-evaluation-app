"""
高端数控机床供应链风险诊断系统（修复版）
模拟场景：某央企五轴机床三大核心部件（数控系统、主轴轴承、光栅尺）的全球供应链
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
# ==================== 1. 构建大规模供应链图谱 ====================
G = nx.DiGraph()

# ---- 国家风险系数 ----
country_risk = {
    '中国': 1, '德国': 4, '日本': 5, '美国': 8,
    '瑞士': 3, '韩国': 4, '意大利': 4, '台湾': 6
}

# ---- 添加节点 (40+) ----
nodes = [
    # 终端客户
    ('GUOJI', {'name': '国机智造', 'country': '中国', 'type': '整机厂'}),

    # ---- 数控系统链路 ----
    ('SIEMENS_CNC', {'name': '西门子数控', 'country': '德国', 'type': '数控系统'}),
    ('FANUC', {'name': '发那科', 'country': '日本', 'type': '数控系统'}),
    ('HEIDENHAIN', {'name': '海德汉', 'country': '德国', 'type': '数控系统'}),
    ('HUAZHONG', {'name': '华中数控', 'country': '中国', 'type': '数控系统'}),
    # 数控系统上游：芯片
    ('XILINX', {'name': '赛灵思', 'country': '美国', 'type': 'FPGA芯片'}),
    ('ALTERA', {'name': '阿尔特拉', 'country': '美国', 'type': 'FPGA芯片'}),
    ('TI', {'name': '德州仪器', 'country': '美国', 'type': 'DSP芯片'}),
    ('RENESAS', {'name': '瑞萨电子', 'country': '日本', 'type': 'MCU芯片'}),
    # 芯片代工
    ('TSMC', {'name': '台积电', 'country': '台湾', 'type': '芯片代工'}),
    ('SAMSUNG_F', {'name': '三星代工', 'country': '韩国', 'type': '芯片代工'}),

    # ---- 主轴轴承链路 ----
    ('SKF', {'name': 'SKF', 'country': '瑞典', 'type': '精密轴承'}),
    ('FAG', {'name': '舍弗勒FAG', 'country': '德国', 'type': '精密轴承'}),
    ('NSK', {'name': 'NSK', 'country': '日本', 'type': '精密轴承'}),
    ('ZWZ', {'name': '瓦轴集团', 'country': '中国', 'type': '精密轴承'}),
    # 轴承上游：特种钢
    ('OVAKO', {'name': '奥钢联', 'country': '奥地利', 'type': '轴承钢'}),
    ('SANYO_SEIKO', {'name': '山阳特钢', 'country': '日本', 'type': '轴承钢'}),
    ('CITIC_SPECIAL', {'name': '中信特钢', 'country': '中国', 'type': '轴承钢'}),

    # ---- 光栅尺/编码器链路 ----
    ('HEIDENHAIN_ENC', {'name': '海德汉编码器', 'country': '德国', 'type': '光栅尺'}),
    ('RENISHAW', {'name': '雷尼绍', 'country': '英国', 'type': '光栅尺'}),
    ('MITUTOYO', {'name': '三丰', 'country': '日本', 'type': '光栅尺'}),
    ('CHANGCHUN_G', {'name': '长春光机所', 'country': '中国', 'type': '光栅尺'}),
    # 光栅尺上游
    ('SCHOTT', {'name': '肖特玻璃', 'country': '德国', 'type': '光学玻璃'}),
    ('OHARA', {'name': '小原光学', 'country': '日本', 'type': '光学玻璃'}),
    ('CDGM', {'name': '成都光明', 'country': '中国', 'type': '光学玻璃'}),
]

G.add_nodes_from(nodes)

# ---- 添加供应关系 (50+) 格式: (上游, 下游, 属性字典) ----
edges = [
    # 数控系统供应
    ('SIEMENS_CNC', 'GUOJI', {'product': '840D系统', 'dependency': 8, 'substitutability': 3}),
    ('FANUC', 'GUOJI', {'product': '31i系统', 'dependency': 7, 'substitutability': 3}),
    ('HEIDENHAIN', 'GUOJI', {'product': 'TNC640', 'dependency': 5, 'substitutability': 4}),
    ('HUAZHONG', 'GUOJI', {'product': '华中8型', 'dependency': 3, 'substitutability': 7}),

    # 芯片供应给数控系统
    ('XILINX', 'SIEMENS_CNC', {'product': 'FPGA', 'dependency': 9, 'substitutability': 2}),
    ('XILINX', 'FANUC', {'product': 'FPGA', 'dependency': 8, 'substitutability': 2}),
    ('ALTERA', 'HEIDENHAIN', {'product': 'FPGA', 'dependency': 7, 'substitutability': 3}),
    ('TI', 'SIEMENS_CNC', {'product': 'DSP', 'dependency': 6, 'substitutability': 3}),
    ('TI', 'FANUC', {'product': 'DSP', 'dependency': 6, 'substitutability': 3}),
    ('RENESAS', 'FANUC', {'product': 'MCU', 'dependency': 5, 'substitutability': 4}),
    ('RENESAS', 'HUAZHONG', {'product': 'MCU', 'dependency': 4, 'substitutability': 5}),

    # 芯片代工
    ('TSMC', 'XILINX', {'product': '7nm晶圆', 'dependency': 10, 'substitutability': 2}),
    ('TSMC', 'ALTERA', {'product': '7nm晶圆', 'dependency': 9, 'substitutability': 2}),
    ('SAMSUNG_F', 'XILINX', {'product': '8nm晶圆', 'dependency': 4, 'substitutability': 5}),
    ('SAMSUNG_F', 'TI', {'product': '45nm晶圆', 'dependency': 5, 'substitutability': 4}),

    # 主轴轴承供应
    ('SKF', 'GUOJI', {'product': 'P2级主轴轴承', 'dependency': 9, 'substitutability': 2}),
    ('FAG', 'GUOJI', {'product': 'P2级主轴轴承', 'dependency': 7, 'substitutability': 3}),
    ('NSK', 'GUOJI', {'product': 'P2级主轴轴承', 'dependency': 5, 'substitutability': 4}),
    ('ZWZ', 'GUOJI', {'product': 'P4级轴承', 'dependency': 3, 'substitutability': 6}),

    # 轴承钢供应
    ('OVAKO', 'SKF', {'product': '高纯净轴承钢', 'dependency': 8, 'substitutability': 3}),
    ('OVAKO', 'FAG', {'product': '高纯净轴承钢', 'dependency': 7, 'substitutability': 3}),
    ('SANYO_SEIKO', 'NSK', {'product': '轴承钢', 'dependency': 6, 'substitutability': 4}),
    ('CITIC_SPECIAL', 'ZWZ', {'product': '轴承钢', 'dependency': 5, 'substitutability': 6}),
    ('CITIC_SPECIAL', 'SKF', {'product': '轴承钢', 'dependency': 3, 'substitutability': 7}),

    # 光栅尺/编码器供应
    ('HEIDENHAIN_ENC', 'GUOJI', {'product': '绝对光栅尺', 'dependency': 8, 'substitutability': 3}),
    ('RENISHAW', 'GUOJI', {'product': '光栅尺', 'dependency': 5, 'substitutability': 4}),
    ('MITUTOYO', 'GUOJI', {'product': '光栅尺', 'dependency': 4, 'substitutability': 5}),
    ('CHANGCHUN_G', 'GUOJI', {'product': '光栅尺', 'dependency': 2, 'substitutability': 7}),

    # 光学玻璃供应
    ('SCHOTT', 'HEIDENHAIN_ENC', {'product': '零膨胀玻璃', 'dependency': 8, 'substitutability': 2}),
    ('OHARA', 'MITUTOYO', {'product': '光学玻璃', 'dependency': 7, 'substitutability': 3}),
    ('CDGM', 'CHANGCHUN_G', {'product': '光学玻璃', 'dependency': 5, 'substitutability': 6}),
    ('CDGM', 'HEIDENHAIN_ENC', {'product': '光学玻璃', 'dependency': 4, 'substitutability': 6}),
]

G.add_edges_from(edges)

# 补充国家风险（奥地利、瑞典、英国）
country_risk.update({'奥地利': 3, '瑞典': 3, '英国': 5})

# ==================== 2. 量化边风险 ====================
for u, v, data in G.edges(data=True):
    dep = data['dependency']
    sub = data['substitutability']
    c_risk = country_risk.get(G.nodes[u]['country'], 5)
    data['risk'] = dep * (11 - sub) * c_risk

# ==================== 3. 风险清单 ====================
edge_risks = []
for u, v, d in G.edges(data=True):
    edge_risks.append({
        '供应商': G.nodes[u]['name'], '客户': G.nodes[v]['name'],
        '产品': d['product'], '依赖度': d['dependency'],
        '可替代性': d['substitutability'], '风险得分': d['risk']
    })
df_risk = pd.DataFrame(edge_risks).sort_values('风险得分', ascending=False)
print("="*60)
print("Top 10 高风险供应关系")
print(df_risk.head(10).to_string(index=False))

# ==================== 4. 节点重要性 ====================
pagerank = nx.pagerank(G, weight='risk')
in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())

node_importance = []
for n in G.nodes():
    importance = pagerank[n]*10 + in_deg[n] + out_deg[n]
    node_importance.append({
        '节点': G.nodes[n]['name'], '国家': G.nodes[n]['country'],
        'PageRank': round(pagerank[n],3), '入度': in_deg[n],
        '出度': out_deg[n], '综合重要性': round(importance,2)
    })
df_node = pd.DataFrame(node_importance).sort_values('综合重要性', ascending=False)
print("\nTop 10 关键节点（综合重要性）")
print(df_node.head(10).to_string(index=False))

# ==================== 5. 关键风险路径 ====================
def find_risk_paths(G, target, max_depth=5):
    """找到以target为终点的上游路径（正向：上游→下游）"""
    paths = []
    def dfs(node, path, depth):
        if depth > max_depth: return
        path.append(node)
        preds = list(G.predecessors(node))
        if not preds:
            paths.append(list(reversed(path)))
        else:
            for pred in preds:
                if pred not in path:
                    dfs(pred, path, depth+1)
        path.pop()
    dfs(target, [], 0)
    return paths

all_paths = find_risk_paths(G, 'GUOJI', max_depth=5)
path_scores = []
for p in all_paths:
    score = sum(G[p[i]][p[i+1]]['risk'] for i in range(len(p)-1))
    path_scores.append({'path': p, 'score': score})

df_path = pd.DataFrame(path_scores).sort_values('score', ascending=False).head(3)
print("\nTop 3 最高风险路径")
for _, row in df_path.iterrows():
    names = [G.nodes[n]['name'] for n in row['path']]
    print(f"风险 {row['score']:.0f}: {' → '.join(names)}")

# ==================== 6. 中断模拟 ====================
print("\n中断传播模拟（无替代的才报告）")
for node in ['TSMC', 'XILINX', 'SKF', 'SCHOTT']:
    affected = []
    for v in G.successors(node):
        prod = G[node][v].get('product','')
        alts = [u for u in G.predecessors(v) if u!=node and G[u][v].get('product')==prod]
        if not alts:
            affected.append(G.nodes[v]['name'])
    if affected:
        print(f"❌ {G.nodes[node]['name']} 中断 → 影响: {', '.join(affected)} (无替代)")

# ==================== 7. 可视化 ====================
fig, ax = plt.subplots(figsize=(18, 14))
pos = nx.kamada_kawai_layout(G)

color_map = {'中国':'#2E7D32', '德国':'#1565C0', '美国':'#C62828',
             '日本':'#F9A825', '台湾':'#EF6C00', '韩国':'#7B1FA2',
             '瑞士':'#00838F', '奥地利':'#4E342E', '瑞典':'#1B5E20', '英国':'#0D47A1'}
node_colors = [color_map.get(G.nodes[n]['country'], '#9E9E9E') for n in G.nodes()]
node_sizes = [pagerank[n]*5000+400 for n in G.nodes()]

edge_colors, edge_widths = [], []
for u,v in G.edges():
    r = G[u][v]['risk']
    edge_widths.append(r/30)
    edge_colors.append('#C62828' if r>200 else '#F9A825' if r>100 else '#4CAF50')

nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7, arrowsize=12)
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, edgecolors='white', linewidths=1.5)
labels = {n: G.nodes[n]['name'] for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=7)

legend_items = [
    Patch(facecolor=color_map[c], label=f'{c} ({country_risk.get(c,5)})') for c in color_map
] + [
    Patch(facecolor='#C62828', alpha=0.6, label='高风险边 (>200)'),
    Patch(facecolor='#F9A825', alpha=0.6, label='中风险边'),
    Patch(facecolor='#4CAF50', alpha=0.6, label='低风险边')
]
ax.legend(handles=legend_items, loc='upper left', fontsize=7, ncol=2)
ax.set_title('五轴数控机床核心部件供应链风险地图\n(节点大小=PageRank重要性，边色/宽=风险)', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()