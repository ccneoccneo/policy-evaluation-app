"""
供应链薄弱环节诊断（修正版）
功能：知识图谱构建 + 节点重要性 + 关键风险路径 + 中断传播模拟
修正：路径方向改为上游→下游，避免KeyError
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
# ==================== 1. 构建供应链知识图谱 ====================
G = nx.DiGraph()

nodes = [
    ('SIEMENS', {'name': '西门子能源', 'country': '德国', 'type': 'OEM'}),
    ('GE', {'name': '通用电气', 'country': '美国', 'type': 'OEM'}),
    ('PCC', {'name': '精密铸件', 'country': '美国', 'type': '叶片铸造'}),
    ('HOWMET', {'name': '豪梅特', 'country': '美国', 'type': '精加工'}),
    ('CISRI', {'name': '钢研总院', 'country': '中国', 'type': '材料研发'}),
    ('AECC', {'name': '中国航发', 'country': '中国', 'type': '叶片制造'}),
    ('WZ', {'name': '瓦轴集团', 'country': '中国', 'type': '轴承'}),
    ('POWER', {'name': '某电力央企', 'country': '中国', 'type': '终端用户'}),
]
G.add_nodes_from(nodes)

edges = [
    # 上游 -> 下游
    ('PCC', 'SIEMENS', {'product': '铸造叶片', 'dependency': 8, 'substitutability': 3}),
    ('HOWMET', 'SIEMENS', {'product': '精加工叶片', 'dependency': 7, 'substitutability': 3}),
    ('HOWMET', 'GE', {'product': '精加工叶片', 'dependency': 7, 'substitutability': 3}),
    ('SIEMENS', 'POWER', {'product': '燃机整机', 'dependency': 9, 'substitutability': 3}),
    ('GE', 'POWER', {'product': '燃机整机', 'dependency': 7, 'substitutability': 4}),
    ('CISRI', 'AECC', {'product': '高温合金', 'dependency': 4, 'substitutability': 7}),
    ('AECC', 'POWER', {'product': '国产叶片', 'dependency': 2, 'substitutability': 9}),
    ('WZ', 'POWER', {'product': '国产轴承', 'dependency': 3, 'substitutability': 8}),
]
G.add_edges_from(edges)

# 国家风险系数
country_risk = {'中国': 1, '德国': 4, '美国': 8, '日本': 5, '瑞士': 3}

# 计算每条边的综合风险得分
for u, v, data in G.edges(data=True):
    dep = data['dependency']
    sub = data['substitutability']
    c_risk = country_risk.get(G.nodes[u]['country'], 5)
    data['risk'] = dep * (11 - sub) * c_risk

# ==================== 2. 节点重要性分析 ====================
in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())
pagerank = nx.pagerank(G, weight='risk')

print("=" * 60)
print("供应链薄弱环节诊断报告")
print("=" * 60)
print("\n--- 节点重要性排名 ---")
importance = []
for node in G.nodes():
    score = pagerank[node] * 10 + (in_deg[node] + out_deg[node])
    importance.append({
        '节点': G.nodes[node]['name'],
        '国家': G.nodes[node]['country'],
        '入度(客户数)': in_deg[node],
        '出度(供应商数)': out_deg[node],
        'PageRank': pagerank[node],
        '综合重要性': round(score, 2)
    })
df_imp = pd.DataFrame(importance).sort_values('综合重要性', ascending=False)
print(df_imp.to_string(index=False))

# ==================== 3. 关键路径分析（风险权重最大路径） ====================
def find_all_paths_to_target(G, target, max_depth=5):
    """
    找出所有从上游供应商到 target 的简单路径（正向：上游→下游）
    在DFS回溯时反转路径得到正向顺序
    """
    all_paths = []
    def dfs(current, path, depth):
        if depth > max_depth:
            return
        path.append(current)
        preds = list(G.predecessors(current))
        if not preds:  # 无前驱，已到源头
            all_paths.append(list(reversed(path)))
        else:
            for pred in preds:
                if pred not in path:   # 防止环路
                    dfs(pred, path, depth + 1)
        path.pop()
    dfs(target, [], 0)
    return all_paths

target = 'POWER'
all_paths = find_all_paths_to_target(G, target, max_depth=4)

path_risks = []
for path in all_paths:
    total_risk = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        total_risk += G[u][v]['risk']
    path_risks.append({'path': path, 'total_risk': total_risk})

df_paths = pd.DataFrame(path_risks).sort_values('total_risk', ascending=False).head(5)
print("\n--- 风险最高路径 Top 5 ---")
for _, row in df_paths.iterrows():
    path_names = [G.nodes[n]['name'] for n in row['path']]
    print(f"风险 {row['total_risk']:.1f}: {' → '.join(path_names)}")

# ==================== 4. 中断传播模拟 ====================
def simulate_disruption(G, disrupted_node):
    """
    模拟某节点中断后，下游客户受影响情况
    检查每个直接下游是否有其他供应商提供相同产品
    """
    affected = []
    for v in G.successors(disrupted_node):
        product = G[disrupted_node][v].get('product', '')
        alternatives = []
        for u in G.predecessors(v):
            if u != disrupted_node and G[u][v].get('product') == product:
                alternatives.append(u)
        if not alternatives:
            affected.append(v)
    return affected

print("\n--- 中断传播模拟 ---")
vulnerable_nodes = []
for node in G.nodes():
    affected = simulate_disruption(G, node)
    if affected:
        affected_names = [G.nodes[a]['name'] for a in affected]
        vulnerable_nodes.append({
            '中断节点': G.nodes[node]['name'],
            '受影响客户': ', '.join(affected_names),
            '受影响数量': len(affected)
        })

df_vuln = pd.DataFrame(vulnerable_nodes).sort_values('受影响数量', ascending=False)
if len(df_vuln) > 0:
    print(df_vuln.to_string(index=False))
else:
    print("所有节点均有替代供应商，系统韧性良好。")

# ==================== 5. 综合薄弱环节清单 ====================
print("\n--- 综合薄弱环节清单（按边风险排序） ---")
edge_risk_list = []
for u, v, data in G.edges(data=True):
    edge_risk_list.append({
        '供应商': G.nodes[u]['name'],
        '客户': G.nodes[v]['name'],
        '产品': data['product'],
        '依赖度': data['dependency'],
        '可替代性': data['substitutability'],
        '风险得分': data['risk']
    })
df_edge = pd.DataFrame(edge_risk_list).sort_values('风险得分', ascending=False).head(5)
print(df_edge[['供应商', '产品', '依赖度', '可替代性', '风险得分']].to_string(index=False))

# 结合节点重要性，输出重点监控对象
print("\n--- 重点监控对象 ---")
high_risk_nodes = df_edge['供应商'].unique()
for node in df_imp.itertuples():
    if node.节点 in high_risk_nodes or node.PageRank > 0.2:
        print(f"⚠️ {node.节点}: 重要性{node.综合重要性:.1f}, 国家{node.国家}, 需重点监控")

# ==================== 6. 可视化 ====================
fig, ax = plt.subplots(figsize=(14, 10))
pos = nx.spring_layout(G, k=3, seed=42)
color_map = {'中国': '#2E7D32', '德国': '#1565C0', '美国': '#C62828'}
node_colors = [color_map.get(G.nodes[n]['country'], '#9E9E9E') for n in G.nodes()]
node_sizes = [pagerank[n]*3000 + 500 for n in G.nodes()]

edge_colors = []
edge_widths = []
for u, v in G.edges():
    risk = G[u][v]['risk']
    edge_widths.append(risk / 30)
    if risk > 200:
        edge_colors.append('#C62828')
    elif risk > 100:
        edge_colors.append('#F9A825')
    else:
        edge_colors.append('#4CAF50')

nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7, arrowsize=15)
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, edgecolors='white', linewidths=2)
labels = {n: G.nodes[n]['name'] for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=9)

from matplotlib.patches import Patch
legend = [
    Patch(facecolor='#2E7D32', label='中国 (低风险)'),
    Patch(facecolor='#1565C0', label='德国 (中风险)'),
    Patch(facecolor='#C62828', label='美国 (高风险)'),
    Patch(facecolor='#C62828', alpha=0.6, label='高风险边 (>200)'),
    Patch(facecolor='#F9A825', alpha=0.6, label='中风险边 (100-200)'),
    Patch(facecolor='#4CAF50', alpha=0.6, label='低风险边 (<100)'),
]
ax.legend(handles=legend, loc='upper left', fontsize=9)
ax.set_title('供应链薄弱环节诊断\n(节点大小=重要性，边色/宽=风险)', fontsize=13)
ax.axis('off')
plt.tight_layout()
plt.show()