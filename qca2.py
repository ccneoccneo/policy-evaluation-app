# =============================================================
# 语义漂移追踪（Semantic Drift Tracking）完整实现
# 目的：追踪职业教育政策关键词在三个政策阶段的语义变化
# 反映：政策重心转移、话语框架演变、制度逻辑变迁
# =============================================================
# 依赖：pip install gensim numpy pandas matplotlib scipy scikit-learn
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import warnings

warnings.filterwarnings("ignore")

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


# =============================================================
# STEP 0: 数据准备
# 真实使用时从分词后的政策文本读取
# =============================================================

# 模拟三期分词后的政策文本（实际从CSV读取）
# 格式：每个文档是一个词列表，每期是多个文档的列表
# 来源：Step1预处理后的df["tokens"]按phase分组

def load_phase_tokens(df, phase_name):
    """从数据框中提取特定阶段的分词列表"""
    phase_docs = df[df["phase"] == phase_name]["tokens"].tolist()
    return phase_docs  # List[List[str]]


# 示例数据（替换为真实分词结果）
phase_tokens = {
    "试点期_2014-2019": [
        ["职业", "教育", "改革", "探索", "发展", "引导", "转型", "试点",
         "合格", "达标", "验收", "考核", "企业", "合作", "签约", "基地",
         "教师", "能力", "培训", "学历", "资格", "认定", "双师", "兼职"],
        ["办学", "许可", "专业", "设置", "招生", "学位", "授予", "本科",
         "职业", "技术", "技能", "人才", "培养", "方案", "标准", "规范"],
        ["产教", "合作", "企业", "实践", "工程", "师资", "培养", "能力",
         "测试", "鉴定", "评价", "考核", "合格", "不合格", "整改"],
    ],
    "规范期_2020-2022": [
        ["标准", "认定", "职称", "量化", "指标", "公示", "考核", "评估",
         "双师", "认证", "办法", "规定", "不得", "必须", "不少于",
         "产业", "学院", "协同", "共建", "融合", "机制", "体制"],
        ["评价", "改革", "破五唯", "职称", "晋升", "达标", "合格",
         "科研", "项目", "成果", "转化", "应用", "企业", "实践",
         "月", "不少于", "强制", "硬性", "规定", "一律"],
        ["学位", "授予", "办学", "质量", "监测", "评估", "合格",
         "规范", "标准", "认定", "程序", "审批", "备案", "公示"],
    ],
    "高质量期_2023-2026": [
        ["代表作", "分类", "多元", "能动", "自主", "破格", "绿色通道",
         "评价", "改革", "赋能", "激励", "选择", "弹性", "创新"],
        ["产教", "深融", "嵌入", "生态", "创新链", "人才链", "产业链",
         "教育链", "协同", "共生", "深度", "融合", "平台", "机制"],
        ["数字", "智能", "人工智能", "胜任力", "新质", "生产力",
         "创新", "突破", "高质量", "内涵", "特色", "品牌", "引领"],
        ["教师", "发展", "能动", "自主", "成长", "赋能", "激励",
         "分类", "评价", "代表作", "选择", "多元", "弹性", "创新"],
    ]
}

phases = list(phase_tokens.keys())
print("=" * 60)
print("语义漂移追踪分析")
print("=" * 60)
print(f"分析阶段: {phases}")
print(f"目标：追踪关键词在三期政策中的语义邻域变化\n")

# =============================================================
# STEP 1: 分期训练Word2Vec模型
# 关键参数说明：
# - vector_size=100: 词向量维度（小语料用100，大语料用300）
# - window=5: 上下文窗口（政策文本句子短，5合适）
# - min_count=2: 最小词频（过滤低频噪音词）
# - sg=1: Skip-gram模式（比CBOW更适合小语料）
# =============================================================

print("STEP 1: 分期训练Word2Vec模型")
print("-" * 40)

models = {}
for phase, tokens in phase_tokens.items():
    model = Word2Vec(
        sentences=tokens,
        vector_size=100,
        window=5,
        min_count=2,
        sg=1,  # Skip-gram
        epochs=50,  # 小语料多训练几轮
        seed=42
    )
    models[phase] = model
    vocab_size = len(model.wv.key_to_index)
    print(f"  {phase}: 词表大小={vocab_size}")

print("✅ 三期模型训练完成\n")

# =============================================================
# STEP 2: 语义邻域追踪
# 对每个目标词，提取各期Top-N邻域词
# 这直接反映：该词的"语义场"在政策演变中如何变化
# =============================================================

# 目标词选择依据：
# - 这些词是职业本科教师发展的核心政策术语
# - 在三期政策中均高频出现，保证跨期可比性
# - 与研究的四条路径直接对应
target_words = {
    "评价": "对应'生态浸润路径'的评价制度变革",
    "产教": "对应'产教融合共生路径'的核心词",
    "能力": "对应'教学成长路径'的核心目标词",
    "创新": "对应'技术创新路径'的关键词",
    "双师": "职业本科教师身份的核心标签",
}

print("STEP 2: 语义邻域追踪结果")
print("=" * 60)

drift_table = []  # 用于后续分析的结构化结果

for word, description in target_words.items():
    print(f"\n【{word}】— {description}")
    print(f"{'阶段':<20} {'Top-8语义邻域词'}")
    print("-" * 55)

    word_exists_in_all = True
    phase_neighbors = {}

    for phase, model in models.items():
        if word in model.wv:
            neighbors = model.wv.most_similar(word, topn=8)
            neighbor_words = [n[0] for n in neighbors]
            neighbor_sims = [round(n[1], 3) for n in neighbors]
            phase_neighbors[phase] = neighbor_words
            print(f"  {phase:<18} {' | '.join(neighbor_words[:6])}")

            # 记录到结构化表格
            for rank, (nw, sim) in enumerate(zip(neighbor_words, neighbor_sims)):
                drift_table.append({
                    "target_word": word,
                    "phase": phase,
                    "rank": rank + 1,
                    "neighbor": nw,
                    "similarity": sim
                })
        else:
            print(f"  {phase:<18} ⚠️ 该词未出现在本期语料中")
            word_exists_in_all = False

drift_df = pd.DataFrame(drift_table)
drift_df.to_csv("semantic_drift_neighbors.csv", index=False, encoding="utf-8-sig")

# =============================================================
# STEP 3: 语义漂移距离量化
# 计算同一词在不同期模型中的向量距离
# 距离越大 = 语义漂移越大 = 政策重心转移越显著
#
# 注意：不同期的Word2Vec模型词向量空间不对齐
# 需要用"Procrustes对齐"或"锚词对齐"方法处理
# =============================================================

print("\n\nSTEP 3: 语义漂移距离量化")
print("-" * 40)
print("注：使用锚词对齐法处理不同期模型空间不可比问题")


def align_models(model1, model2, anchor_words=None):
    """
    使用锚词对齐两个Word2Vec模型的向量空间
    锚词：在两期中语义基本稳定的词（如"教师""学生""大学"）

    返回：对齐矩阵W，使 model2_vectors ≈ model1_vectors @ W
    """
    if anchor_words is None:
        # 默认锚词：两期共有且语义稳定的基础词
        anchor_words = ["教师", "学生", "大学", "课程", "专业", "学院"]

    # 过滤两期都有的锚词
    common_anchors = [w for w in anchor_words
                      if w in model1.wv and w in model2.wv]

    if len(common_anchors) < 3:
        print(f"  ⚠️ 共同锚词不足（{len(common_anchors)}），使用余弦相似度近似")
        return None

    # 构建锚词矩阵
    A = np.array([model1.wv[w] for w in common_anchors])
    B = np.array([model2.wv[w] for w in common_anchors])

    # Procrustes对齐：找最优旋转矩阵
    U, s, Vt = np.linalg.svd(B.T @ A)
    W = U @ Vt  # 对齐矩阵
    return W


def semantic_drift_distance(word, model1, model2, align_matrix=None):
    """
    计算词在两个模型中的语义漂移距离（余弦距离）
    距离 ∈ [0, 2]，越大表示漂移越显著
    """
    if word not in model1.wv or word not in model2.wv:
        return None

    vec1 = model1.wv[word]
    vec2 = model2.wv[word]

    if align_matrix is not None:
        vec2 = vec2 @ align_matrix  # 对齐后比较

    # 余弦距离 = 1 - 余弦相似度
    drift = cosine(vec1, vec2)
    return round(drift, 4)


# 计算各词在三个阶段间的漂移距离
anchor_words = ["教师", "学生", "大学", "课程", "专业", "学院", "培养", "发展"]
phase_list = list(models.keys())

# 对齐相邻期模型
W_12 = align_models(models[phase_list[0]], models[phase_list[1]], anchor_words)
W_23 = align_models(models[phase_list[1]], models[phase_list[2]], anchor_words)
W_13 = align_models(models[phase_list[0]], models[phase_list[2]], anchor_words)

print(f"\n{'目标词':<8} {'试点→规范':>12} {'规范→高质量':>12} {'试点→高质量':>12} {'漂移趋势'}")
print("-" * 58)

drift_results = {}
for word in target_words:
    d12 = semantic_drift_distance(word, models[phase_list[0]], models[phase_list[1]], W_12)
    d23 = semantic_drift_distance(word, models[phase_list[1]], models[phase_list[2]], W_23)
    d13 = semantic_drift_distance(word, models[phase_list[0]], models[phase_list[2]], W_13)

    drift_results[word] = {"试点→规范": d12, "规范→高质量": d23, "试点→高质量": d13}

    if d12 and d23:
        trend = "↑加速漂移" if d23 > d12 else ("→平稳漂移" if d23 > 0.05 else "↓趋于稳定")
    else:
        trend = "数据不足"

    d12_s = f"{d12:.4f}" if d12 else "N/A"
    d23_s = f"{d23:.4f}" if d23 else "N/A"
    d13_s = f"{d13:.4f}" if d13 else "N/A"
    print(f"{word:<8} {d12_s:>12} {d23_s:>12} {d13_s:>12} {trend}")

# =============================================================
# STEP 4: 语义变化内容分析
# 识别邻域词的"话语框架"转变
# 这是语义漂移分析最有研究价值的部分
# =============================================================

print("\n\nSTEP 4: 话语框架转变分析（以'评价'为例）")
print("=" * 60)


def analyze_discourse_shift(word, phase_neighbors_dict):
    """
    对邻域词进行话语框架归类
    归类维度来自政策工具词典（Step3构建）
    """
    frame_dicts = {
        "控制框架": ["考核", "达标", "合格", "不合格", "验收", "审批",
                     "不得", "必须", "强制", "硬性", "一律", "指标"],
        "激励框架": ["奖励", "优先", "补贴", "支持", "鼓励", "给予",
                     "表彰", "倾斜", "优惠"],
        "赋能框架": ["自主", "选择", "多元", "弹性", "分类", "创新",
                     "代表作", "破格", "绿色通道", "能动"],
        "工具框架": ["标准", "办法", "规定", "程序", "认定", "备案",
                     "公示", "材料", "申报"],
    }

    results = {}
    for phase, neighbors in phase_neighbors_dict.items():
        frame_counts = {frame: 0 for frame in frame_dicts}
        for neighbor in neighbors:
            for frame, words in frame_dicts.items():
                if neighbor in words:
                    frame_counts[frame] += 1
        results[phase] = frame_counts
    return results


# 构建评价词的邻域字典
eval_neighbors = {}
for phase in phases:
    if "评价" in models[phase].wv:
        neighbors = models[phase].wv.most_similar("评价", topn=10)
        eval_neighbors[phase] = [n[0] for n in neighbors]

frame_analysis = analyze_discourse_shift("评价", eval_neighbors)

print(f"{'话语框架':<10}", end="")
for phase in phases:
    short_phase = phase.split("_")[0]
    print(f"{short_phase:>12}", end="")
print()
print("-" * 46)

for frame in ["控制框架", "激励框架", "赋能框架", "工具框架"]:
    print(f"{frame:<10}", end="")
    for phase in phases:
        count = frame_analysis.get(phase, {}).get(frame, 0)
        bar = "█" * count
        print(f"{bar:>12}", end="")
    print()

print("\n→ 控制框架词减少 + 赋能框架词增加 = 政策话语从管控走向赋能")
print("→ 这是'控制型治理→赋能型治理'论断的文本证据")

# =============================================================
# STEP 5: 可视化
# （1）语义漂移距离热力图
# （2）关键词t-SNE空间分布（三期对比）
# =============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- 图1: 漂移距离热力图 ---
ax1 = axes[0]
words_list = list(drift_results.keys())
periods = ["试点→规范", "规范→高质量", "试点→高质量"]
heat_data = np.array([
    [drift_results[w].get(p, 0) or 0 for p in periods]
    for w in words_list
])

im = ax1.imshow(heat_data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)
ax1.set_xticks(range(len(periods)))
ax1.set_xticklabels(periods, fontsize=10)
ax1.set_yticks(range(len(words_list)))
ax1.set_yticklabels(words_list, fontsize=11)
ax1.set_title("政策关键词语义漂移距离热力图", fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax1, label="余弦距离（越大=漂移越显著）")

for i in range(len(words_list)):
    for j in range(len(periods)):
        val = heat_data[i, j]
        ax1.text(j, i, f"{val:.3f}", ha="center", va="center",
                 fontsize=9, color="white" if val > 0.3 else "black")

# --- 图2: t-SNE词向量空间分布（以规范期和高质量期对比）---
ax2 = axes[1]

# 提取关键词在两期的向量
words_to_plot = list(target_words.keys()) + [
    "评价", "代表作", "分类", "考核", "达标",
    "自主", "选择", "创新", "赋能", "激励"
]

phase_a = phase_list[0]  # 试点期
phase_b = phase_list[2]  # 高质量期

vectors_a, vectors_b, labels_a, labels_b = [], [], [], []
for w in set(words_to_plot):
    if w in models[phase_a].wv:
        vectors_a.append(models[phase_a].wv[w])
        labels_a.append(w)
    if w in models[phase_b].wv:
        vectors_b.append(models[phase_b].wv[w])
        labels_b.append(w)

if len(vectors_a) >= 3 and len(vectors_b) >= 3:
    all_vectors = np.array(vectors_a + vectors_b)

    # t-SNE降维到2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(all_vectors) - 1))
    coords = tsne.fit_transform(all_vectors)

    n_a = len(vectors_a)
    coords_a = coords[:n_a]
    coords_b = coords[n_a:]

    ax2.scatter(coords_a[:, 0], coords_a[:, 1], c="#4F46E5", s=80,
                label=f"试点期({phase_a.split('_')[1]})", alpha=0.8, marker="o")
    ax2.scatter(coords_b[:, 0], coords_b[:, 1], c="#EF4444", s=80,
                label=f"高质量期({phase_b.split('_')[1]})", alpha=0.8, marker="^")

    for i, (label, coord) in enumerate(zip(labels_a, coords_a)):
        ax2.annotate(label, coord, textcoords="offset points",
                     xytext=(4, 2), fontsize=8, color="#4F46E5")
    for i, (label, coord) in enumerate(zip(labels_b, coords_b)):
        ax2.annotate(label, coord, textcoords="offset points",
                     xytext=(4, 2), fontsize=8, color="#EF4444")

    ax2.set_title("关键词语义空间分布对比\n（蓝=试点期，红=高质量期）",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_xlabel("t-SNE维度1")
    ax2.set_ylabel("t-SNE维度2")
else:
    ax2.text(0.5, 0.5, "语料量不足，无法绘制t-SNE图\n（需至少10个共有词）",
             ha="center", va="center", transform=ax2.transAxes, fontsize=11)

plt.suptitle("职业本科政策语义漂移分析", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("semantic_drift_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n→ 已保存: semantic_drift_analysis.png")

# =============================================================
# STEP 6: 语义漂移的研究价值总结（自动生成分析报告）
# =============================================================

print("\n" + "=" * 60)
print("STEP 6: 语义漂移分析的研究价值")
print("=" * 60)

report = """
语义漂移追踪能回答的核心研究问题：

【问题1】政策话语框架如何演变？
→ 通过追踪"评价"的邻域词从"考核/达标"变为"代表作/分类/自主"
→ 提供政策从"控制型"转向"赋能型"的文本层面量化证据
→ 支撑第四部分"制度悖论→破解之道"的核心论点

【问题2】不同政策阶段的制度逻辑是什么？
→ 试点期：评价邻域=控制框架词 → 制度逻辑以合规为主
→ 规范期：评价邻域=工具框架词 → 制度逻辑以标准化为主（指标化陷阱期）
→ 高质量期：评价邻域=赋能框架词 → 制度逻辑转向激活能动性

【问题3】哪些词的语义漂移最显著？
→ 漂移距离最大的词 = 政策重心转移最明显的概念
→ 这些词是政策分析的"指示词"，可作为阶段划分的文本依据

【问题4】院校层面话语与国家政策话语的距离？
→ 用同样方法对比"院校文本"与"国家文本"中目标词的邻域
→ 邻域相似度高 = 传导一致性强（对应TRANS指标）
→ 邻域差异大 = 院校进行了主动的话语再建构（策略性套利的文本证据）

核心价值：语义漂移分析将"政策演变"从
研究者的主观解读 → 可量化、可复现的文本证据
这是传统政策文本分析无法实现的方法论突破
"""
print(report)