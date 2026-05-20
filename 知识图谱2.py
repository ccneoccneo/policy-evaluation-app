"""
航空发动机高压涡轮单晶叶片供应链风险诊断系统
客户：中国航发集团
产品：单晶涡轮叶片（用于WS/CJ系列发动机）
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
# ==================== 1. 构建供应链图谱 ====================
G = nx.DiGraph()

# ---- 国家风险系数 ----
country_risk = {
    '中国': 1, '美国': 8, '日本': 5, '德国': 4,
    '法国': 4, '英国': 5, '俄罗斯': 6, '哈萨克斯坦': 6,
    '韩国': 4, '瑞典': 3, '荷兰': 6, '以色列': 6
}

# ---- 添加节点 (50+) ----
nodes = [
    # === 终端客户 ===
    ('AECC', {'name': '中国航发(主机厂)', 'country': '中国', 'type': '发动机总成'}),

    # === 叶片制造（一级供应商） ===
    ('AECC_BLADE', {'name': '航发叶片事业部', 'country': '中国', 'type': '叶片制造'}),
    ('PCC_BLADE', {'name': '精密铸件PCC', 'country': '美国', 'type': '叶片铸造'}),
    ('HOWMET', {'name': '豪梅特航空', 'country': '美国', 'type': '叶片精加工'}),
    ('IHI', {'name': '石川岛播磨', 'country': '日本', 'type': '叶片制造'}),

    # === 母合金供应商（二级） ===
    ('CANNON_MUSK', {'name': '卡本特技术', 'country': '美国', 'type': '镍基母合金'}),
    ('HITCHINER', {'name': '希钦纳制造', 'country': '美国', 'type': '母合金'}),
    ('ATI_METALS', {'name': 'ATI特种材料', 'country': '美国', 'type': '母合金'}),
    ('NIPPON_STEEL', {'name': '日本制铁', 'country': '日本', 'type': '耐热合金'}),
    ('BAOSTEEL', {'name': '宝钢特钢', 'country': '中国', 'type': '母合金(试制)'}),
    ('CISRI_ALLOY', {'name': '钢研总院', 'country': '中国', 'type': '母合金研发'}),

    # === 铼金属供应商（战略稀缺材料） ===
    ('MOLYMET', {'name': '智利莫利迈特', 'country': '智利', 'type': '铼金属'}),
    ('FREEPORT', {'name': '美国自由港', 'country': '美国', 'type': '铼金属'}),
    ('KAZ_RHENIUM', {'name': '哈萨克铼业', 'country': '哈萨克斯坦', 'type': '铼金属'}),
    ('JX_METALS', {'name': 'JX金属', 'country': '日本', 'type': '高纯铼'}),
    ('MOLY_CORP', {'name': '洛阳钼业', 'country': '中国', 'type': '铼(伴生回收)'}),

    # === 陶瓷型芯供应商 ===
    ('MORGAN_TC', {'name': '摩根先进材料', 'country': '英国', 'type': '陶瓷型芯'}),
    ('COORSTEK', {'name': 'CoorsTek', 'country': '美国', 'type': '陶瓷型芯'}),
    ('CUMI', {'name': '印度CUMI', 'country': '印度', 'type': '陶瓷型芯'}),
    ('SHANDONG_CERAMIC', {'name': '山东工业陶瓷院', 'country': '中国', 'type': '陶瓷型芯'}),

    # === 涂层材料供应商 ===
    ('PRAXAIR_TA', {'name': '普莱克斯表面技术', 'country': '美国', 'type': '热障涂层'}),
    ('OERLIKON', {'name': '欧瑞康美科', 'country': '瑞士', 'type': '热障涂层'}),
    ('TOCALO', {'name': '东华隆', 'country': '日本', 'type': '涂层加工'}),
    ('AECC_COATING', {'name': '航发涂层中心', 'country': '中国', 'type': '涂层研发'}),

    # === 单晶铸造炉设备商 ===
    ('ALD', {'name': 'ALD真空工业', 'country': '德国', 'type': '单晶炉'}),
    ('CONSARC', {'name': '康萨克', 'country': '美国', 'type': '单晶炉'}),
    ('ECM', {'name': 'ECM Technologies', 'country': '法国', 'type': '单晶炉'}),
    ('AVIC_MANUFACTURING', {'name': '中航制造院', 'country': '中国', 'type': '单晶炉(国产化)'}),
]

G.add_nodes_from(nodes)

# 补充国家风险
country_risk.update({'智利': 3, '印度': 4, '瑞士': 3, '法国': 4})

# ---- 添加供应关系 (70+) 格式: (上游, 下游, 属性字典) ----
edges = [
    # 叶片供应给整机厂
    ('AECC_BLADE', 'AECC', {'product': 'WS/CJ单晶叶片', 'dependency': 6, 'substitutability': 4}),
    ('PCC_BLADE', 'AECC', {'product': '单晶叶片毛坯', 'dependency': 7, 'substitutability': 3}),
    ('HOWMET', 'AECC', {'product': '精加工叶片', 'dependency': 5, 'substitutability': 3}),
    ('IHI', 'AECC', {'product': '低压叶片', 'dependency': 3, 'substitutability': 5}),

    # 母合金供应给叶片制造商
    ('CANNON_MUSK', 'PCC_BLADE', {'product': 'CMSX-4母合金', 'dependency': 9, 'substitutability': 2}),
    ('CANNON_MUSK', 'HOWMET', {'product': '母合金', 'dependency': 8, 'substitutability': 2}),
    ('HITCHINER', 'PCC_BLADE', {'product': '母合金', 'dependency': 4, 'substitutability': 3}),
    ('ATI_METALS', 'HOWMET', {'product': '母合金', 'dependency': 6, 'substitutability': 3}),
    ('NIPPON_STEEL', 'IHI', {'product': '耐热合金', 'dependency': 7, 'substitutability': 3}),
    ('BAOSTEEL', 'AECC_BLADE', {'product': '母合金(国产试制)', 'dependency': 4, 'substitutability': 6}),
    ('CISRI_ALLOY', 'AECC_BLADE', {'product': '研发级母合金', 'dependency': 3, 'substitutability': 7}),

    # 铼金属供应给母合金厂商
    ('MOLYMET', 'CANNON_MUSK', {'product': '高纯铼粒', 'dependency': 8, 'substitutability': 3}),
    ('FREEPORT', 'ATI_METALS', {'product': '铼制品', 'dependency': 7, 'substitutability': 2}),
    ('KAZ_RHENIUM', 'CANNON_MUSK', {'product': '铼酸铵', 'dependency': 5, 'substitutability': 4}),
    ('JX_METALS', 'NIPPON_STEEL', {'product': '高纯铼', 'dependency': 6, 'substitutability': 3}),
    ('MOLY_CORP', 'CISRI_ALLOY', {'product': '回收铼', 'dependency': 3, 'substitutability': 7}),
    ('MOLY_CORP', 'BAOSTEEL', {'product': '回收铼', 'dependency': 2, 'substitutability': 8}),

    # 陶瓷型芯供应给叶片制造商
    ('MORGAN_TC', 'PCC_BLADE', {'product': '陶瓷型芯', 'dependency': 9, 'substitutability': 2}),
    ('MORGAN_TC', 'HOWMET', {'product': '陶瓷型芯', 'dependency': 8, 'substitutability': 2}),
    ('COORSTEK', 'PCC_BLADE', {'product': '陶瓷型芯', 'dependency': 5, 'substitutability': 3}),
    ('CUMI', 'AECC_BLADE', {'product': '陶瓷型芯', 'dependency': 3, 'substitutability': 5}),
    ('SHANDONG_CERAMIC', 'AECC_BLADE', {'product': '国产陶瓷型芯', 'dependency': 2, 'substitutability': 7}),

    # 涂层供应
    ('PRAXAIR_TA', 'PCC_BLADE', {'product': '热障涂层', 'dependency': 8, 'substitutability': 2}),
    ('PRAXAIR_TA', 'HOWMET', {'product': '热障涂层', 'dependency': 8, 'substitutability': 2}),
    ('OERLIKON', 'HOWMET', {'product': '热障涂层', 'dependency': 5, 'substitutability': 3}),
    ('TOCALO', 'IHI', {'product': '涂层加工', 'dependency': 6, 'substitutability': 4}),
    ('AECC_COATING', 'AECC_BLADE', {'product': '国产涂层(研发)', 'dependency': 3, 'substitutability': 6}),

    # 单晶炉设备供应给叶片制造商
    ('ALD', 'PCC_BLADE', {'product': '单晶定向凝固炉', 'dependency': 10, 'substitutability': 1}),
    ('ALD', 'HOWMET', {'product': '单晶定向凝固炉', 'dependency': 9, 'substitutability': 1}),
    ('CONSARC', 'HOWMET', {'product': '单晶炉', 'dependency': 4, 'substitutability': 2}),
    ('ECM', 'IHI', {'product': '单晶炉', 'dependency': 6, 'substitutability': 3}),
    ('AVIC_MANUFACTURING', 'AECC_BLADE', {'product': '国产单晶炉', 'dependency': 3, 'substitutability': 5}),
]

G.add_edges_from(edges)

# ==================== 2. 量化边风险 ====================
for u, v, data in G.edges(data=True):
    dep = data['dependency']
    sub = data['substitutability']
    c_risk = country_risk.get(G.nodes[u]['country'], 5)
    data['risk'] = dep * (11 - sub) * c_risk

# ==================== 3. 生成风险清单 ====================
edge_risks = []
for u, v, d in G.edges(data=True):
    edge_risks.append({
        '上游供应商': G.nodes[u]['name'],
        '下游客户': G.nodes[v]['name'],
        '供应产品': d['product'],
        '依赖度': d['dependency'],
        '可替代性': d['substitutability'],
        '供应国风险系数': country_risk.get(G.nodes[u]['country'], 5),
        '综合风险得分': d['risk']
    })
df_risk = pd.DataFrame(edge_risks).sort_values('综合风险得分', ascending=False)
print("="*70)
print("Top 15 高风险供应关系")
print("="*70)
print(df_risk.head(15).to_string(index=False))

# ==================== 4. 节点重要性 ====================
pagerank = nx.pagerank(G, weight='risk')
in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())

node_imp = []
for n in G.nodes():
    node_imp.append({
        '节点名称': G.nodes[n]['name'],
        '国家': G.nodes[n]['country'],
        '类型': G.nodes[n]['type'],
        'PageRank': round(pagerank[n], 4),
        '下游客户数(出度)': out_deg[n],
        '上游供应商数(入度)': in_deg[n],
        '综合重要性得分': round(pagerank[n]*10 + out_deg[n] + in_deg[n], 2)
    })
df_node = pd.DataFrame(node_imp).sort_values('综合重要性得分', ascending=False)
print("\n" + "="*70)
print("Top 15 关键节点（按PageRank+度中心性综合重要度排序）")
print("="*70)
print(df_node.head(15).to_string(index=False))

# ==================== 5. 关键风险路径 ====================
def find_all_risk_paths(G, target, max_depth=6):
    paths = []
    def dfs(node, path, depth):
        if depth > max_depth:
            return
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

all_paths = find_all_risk_paths(G, 'AECC', max_depth=6)
path_scores = []
for p in all_paths:
    score = sum(G[p[i]][p[i+1]]['risk'] for i in range(len(p)-1))
    path_scores.append({'path': p, 'score': score})

df_path = pd.DataFrame(path_scores).sort_values('score', ascending=False).head(5)
print("\n" + "="*70)
print("Top 5 风险最高的供应路径（上游→下游）")
print("="*70)
for _, row in df_path.iterrows():
    names = [G.nodes[n]['name'] for n in row['path']]
    print(f"风险总分 {row['score']:.0f}: {' → '.join(names)}")

# ==================== 6. 中断传播模拟 ====================
print("\n" + "="*70)
print("中断传播模拟：关键节点断供后的上下游影响")
print("="*70)
key_nodes_to_test = ['ALD', 'CANNON_MUSK', 'PRAXAIR_TA', 'MORGAN_TC', 'MOLYMET']
for node in key_nodes_to_test:
    affected_direct = []
    affected_indirect = set()
    for v in G.successors(node):
        prod = G[node][v].get('product','')
        alts = [u for u in G.predecessors(v) if u!=node and G[u][v].get('product')==prod]
        if not alts:
            affected_direct.append(G.nodes[v]['name'])
            # 继续向下追溯
            for w in G.successors(v):
                affected_indirect.add(G.nodes[w]['name'])
    if affected_direct:
        indirect_list = list(affected_indirect)
        print(f"❌ {G.nodes[node]['name']}({G.nodes[node]['country']}) 中断")
        print(f"   → 直接影响(无替代): {', '.join(affected_direct)}")
        if indirect_list:
            print(f"   → 间接波及: {', '.join(indirect_list)}")
        print()

# ==================== 7. 可视化 ====================
fig, ax = plt.subplots(figsize=(20, 16))
pos = nx.kamada_kawai_layout(G)

color_map = {
    '中国': '#2E7D32', '美国': '#C62828', '日本': '#F9A825',
    '德国': '#1565C0', '英国': '#0D47A1', '法国': '#1E88E5',
    '瑞典': '#1B5E20', '瑞士': '#00838F', '哈萨克斯坦': '#BF360C',
    '智利': '#4E342E', '印度': '#FF8F00', '韩国': '#7B1FA2',
    '荷兰': '#6A1B9A', '以色列': '#0277BD'
}
node_colors = [color_map.get(G.nodes[n]['country'], '#9E9E9E') for n in G.nodes()]
node_sizes = [pagerank[n]*4000+500 for n in G.nodes()]

edge_colors, edge_widths = [], []
for u,v in G.edges():
    r = G[u][v]['risk']
    edge_widths.append(max(0.5, r/50))
    if r > 500:
        edge_colors.append('#B71C1C')
    elif r > 250:
        edge_colors.append('#F9A825')
    elif r > 100:
        edge_colors.append('#43A047')
    else:
        edge_colors.append('#90A4AE')

nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.65, arrowsize=14)
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, edgecolors='white', linewidths=1.8)
labels = {n: G.nodes[n]['name'] for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=6.5)

legend_items = [
    Patch(facecolor=color_map[c], label=f'{c} (风险:{country_risk.get(c,5)})')
    for c in ['中国','美国','日本','德国','英国','哈萨克斯坦','智利']
] + [
    Patch(facecolor='#B71C1C', alpha=0.6, label='极端风险边 (>500)'),
    Patch(facecolor='#F9A825', alpha=0.6, label='高风险边 (250-500)'),
    Patch(facecolor='#43A047', alpha=0.6, label='中风险边 (100-250)'),
]
ax.legend(handles=legend_items, loc='upper left', fontsize=8, ncol=2, framealpha=0.9)
ax.set_title('航空发动机单晶涡轮叶片全球供应链风险地图\n(中国航发视角 · 节点大小=PageRank重要性 · 边色/宽=风险)',
             fontsize=15, fontweight='bold', pad=20)
ax.axis('off')
plt.tight_layout()
plt.show()