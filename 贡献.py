# ============================================================
# 贡献一：政策文本智能分析工具
# 目标：爬取政策文本 → 主题建模 → 输出演变热力图
# ============================================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import jieba
import jieba.analyse
from gensim import corpora, models
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import re

# --- Step 1: 数据采集（模拟，实际替换为真实爬虫） ---
def fetch_policy_texts(urls: list[str]) -> list[dict]:
    """爬取政策文本，返回[{year, title, content}]"""
    results = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.encoding = 'utf-8'
            soup = BeautifulSoup(r.text, 'html.parser')
            title = soup.find('title').text.strip()
            content = ' '.join([p.text for p in soup.find_all('p')])
            year = re.search(r'(20\d{2})', url)
            results.append({
                'year': int(year.group()) if year else 2020,
                'title': title,
                'content': content
            })
        except Exception as e:
            print(f"爬取失败: {url}, 错误: {e}")
    return results

# --- Step 2: 文本清洗与分词 ---
# 添加装备制造业自定义词典
custom_words = ['工业母机', '高端数控机床', '卡脖子', '制造2025', '强链补链',
                '产业链安全', '双碳', '工业软件', '智能制造', '新能源汽车']
for w in custom_words:
    jieba.add_word(w)

stopwords = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
                 '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
                 '你', '会', '着', '没有', '看', '好', '自己', '这'])

def clean_and_tokenize(text: str) -> list[str]:
    text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)  # 只保留中文
    words = jieba.lcut(text)
    return [w for w in words if w not in stopwords and len(w) > 1]

# --- Step 3: LDA主题建模 ---
def lda_topic_modeling(docs: list[str], num_topics: int = 10):
    tokenized = [clean_and_tokenize(doc) for doc in docs]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
    lda = models.LdaModel(
        corpus, num_topics=num_topics,
        id2word=dictionary, passes=20,
        random_state=42
    )
    return lda, corpus, dictionary

# --- Step 4: BERTopic主题建模（更精准） ---
def bertopic_modeling(docs: list[str]):
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    topic_model = BERTopic(
        embedding_model=embedding_model,
        language='chinese (simplified)',
        calculate_probabilities=True,
        verbose=True
    )
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs

# --- Step 5: 绘制政策主题演变热力图 ---
def plot_topic_heatmap(df: pd.DataFrame, topic_cols: list[str]):
    """
    df: 包含year列和各主题占比列
    topic_cols: 主题列名列表
    """
    pivot = df.groupby('year')[topic_cols].mean()
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        pivot.T,
        cmap='YlOrRd',
        annot=True, fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': '主题强度'}
    )
    plt.title('装备制造业政策主题演变热力图（2005-2025）', fontsize=14)
    plt.xlabel('年份')
    plt.ylabel('政策主题')
    plt.tight_layout()
    plt.savefig('policy_topic_heatmap.png', dpi=300)
    plt.show()
    print("热力图已保存为 policy_topic_heatmap.png")


# ============================================================
# 贡献二：产业政策效果评估模型
# 目标：DID + 合成控制法 + 双重机器学习 评估政策净效应
# ============================================================

import numpy as np
from sklearn.preprocessing import StandardScaler
from linearmodels.panel import PanelOLS
import statsmodels.formula.api as smf

# --- Step 1: 构造面板数据（模拟数据，实际从CSMAR/Wind导入） ---
def generate_panel_data(n_firms=200, n_years=10, policy_year=2018,
                        treat_ratio=0.4, true_effect=0.15):
    """生成模拟企业面板数据"""
    np.random.seed(42)
    firms = range(n_firms)
    years = range(2015, 2015 + n_years)
    treated = np.random.choice([0,1], n_firms, p=[1-treat_ratio, treat_ratio])

    records = []
    for i in firms:
        firm_fe = np.random.normal(0, 0.5)  # 企业固定效应
        for y in years:
            post = int(y >= policy_year)
            # 结果变量：研发投入（取对数）
            rd = (2 + firm_fe + 0.05*(y-2015)
                  + true_effect * treated[i] * post
                  + np.random.normal(0, 0.1))
            records.append({
                'firm_id': i,
                'year': y,
                'treated': treated[i],
                'post': post,
                'did': treated[i] * post,
                'rd_log': rd,
                'size': np.random.normal(5, 1),       # 控制变量：企业规模
                'leverage': np.random.uniform(0.1, 0.8)  # 资产负债率
            })
    return pd.DataFrame(records)

# --- Step 2: 平行趋势检验（事件研究法） ---
def parallel_trend_test(df: pd.DataFrame):
    """事件研究法：检验DID前提假设"""
    df = df.copy()
    base_year = df['year'].min()
    # 生成每年的交互项
    for y in df['year'].unique():
        df[f'treat_y{y}'] = df['treated'] * (df['year'] == y).astype(int)

    treat_year_cols = [c for c in df.columns if c.startswith('treat_y')]
    # 去掉政策前一年作为基准
    base_col = f'treat_y{df["year"].min() + 1}'
    cols_to_use = [c for c in treat_year_cols if c != base_col]

    formula = f"rd_log ~ {' + '.join(cols_to_use)} + size + leverage + C(year)"
    model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['firm_id']})

    # 绘制平行趋势图
    coefs = {int(c.replace('treat_y','')): model.params[c] for c in cols_to_use}
    conf = {int(c.replace('treat_y','')): model.conf_int().loc[c] for c in cols_to_use}

    years_sorted = sorted(coefs.keys())
    coef_vals = [coefs[y] for y in years_sorted]
    ci_low = [conf[y][0] for y in years_sorted]
    ci_high = [conf[y][1] for y in years_sorted]

    plt.figure(figsize=(10, 5))
    plt.plot(years_sorted, coef_vals, 'o-', color='steelblue', label='政策效应系数')
    plt.fill_between(years_sorted, ci_low, ci_high, alpha=0.2, color='steelblue')
    plt.axvline(x=2018, color='red', linestyle='--', label='政策实施年份')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    plt.xlabel('年份')
    plt.ylabel('系数估计值')
    plt.title('平行趋势检验（事件研究法）')
    plt.legend()
    plt.tight_layout()
    plt.savefig('parallel_trend.png', dpi=300)
    plt.show()
    return model

# --- Step 3: 双重差分（DID）主模型 ---
def run_did(df: pd.DataFrame):
    """带双向固定效应的DID估计"""
    df_indexed = df.set_index(['firm_id', 'year'])
    # 双向固定效应：控制企业和年份
    model = PanelOLS.from_formula(
        'rd_log ~ did + size + leverage + TimeEffects',
        data=df_indexed,
        entity_effects=True
    )
    res = model.fit(cov_type='clustered', cluster_entity=True)
    print("=" * 50)
    print("DID估计结果（双向固定效应）")
    print("=" * 50)
    print(res.summary.tables[1])
    return res

# --- Step 4: 安慰剂检验 ---
def placebo_test(df: pd.DataFrame, n_iter=500):
    """随机打乱处理组，重复估计，验证真实效应显著性"""
    placebo_effects = []
    for _ in range(n_iter):
        df_pla = df.copy()
        df_pla['treated'] = np.random.permutation(df_pla['treated'].values)
        df_pla['did'] = df_pla['treated'] * df_pla['post']
        try:
            res = smf.ols('rd_log ~ did + size + leverage + C(year) + C(firm_id)',
                          data=df_pla).fit()
            placebo_effects.append(res.params['did'])
        except:
            continue

    real_effect = 0.148  # 替换为真实DID系数
    plt.figure(figsize=(8, 5))
    plt.hist(placebo_effects, bins=50, color='steelblue', alpha=0.7, label='安慰剂效应分布')
    plt.axvline(x=real_effect, color='red', linestyle='--', linewidth=2, label=f'真实效应={real_effect:.3f}')
    plt.xlabel('估计系数')
    plt.ylabel('频次')
    plt.title('安慰剂检验')
    plt.legend()
    plt.tight_layout()
    plt.savefig('placebo_test.png', dpi=300)
    plt.show()
    p_value = np.mean(np.abs(placebo_effects) >= abs(real_effect))
    print(f"安慰剂检验 p值 = {p_value:.4f}（<0.05说明真实效应显著）")

# --- Step 5: 双重机器学习（DML）—— 处理高维控制变量 ---
from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

def run_dml(df: pd.DataFrame):
    """用双重机器学习估计异质性处理效应"""
    X = df[['size', 'leverage']].values
    Y = df['rd_log'].values
    T = df['did'].values

    est = LinearDML(
        model_y=GradientBoostingRegressor(n_estimators=100),
        model_t=GradientBoostingClassifier(n_estimators=100),
        random_state=42
    )
    est.fit(Y, T, X=X)
    te = est.effect(X)
    print(f"\nDML估计 | 平均处理效应(ATE): {te.mean():.4f} | 标准差: {te.std():.4f}")
    return est, te


# ============================================================
# 贡献三：500强多维评价（精简版，聚焦文本CSR分析）
# 目标：用大模型分析年报ESG内容，生成CSR得分
# ============================================================

from transformers import pipeline

def compute_csr_score(annual_report_text: str) -> dict:
    """
    用零样本分类分析年报ESG内容
    返回各维度得分
    """
    # 用Huggingface零样本分类器（无需标注数据）
    classifier = pipeline(
        "zero-shot-classification",
        model="IDEA-CCNL/Ernie-Zeus-zeroshot-nli-cn",  # 中文零样本模型
        device=-1  # CPU, 如有GPU改为0
    )

    # ESG维度定义
    esr_dims = {
        "环境保护": ["节能减排", "碳中和", "绿色制造", "环境治理", "可持续发展"],
        "员工权益": ["员工培训", "薪酬福利", "劳动保护", "人才发展"],
        "社区贡献": ["公益捐赠", "扶贫攻坚", "乡村振兴", "社区服务"],
        "公司治理": ["信息披露", "独立董事", "股东权益", "合规经营"]
    }

    # 将年报切分为段落（每段不超过512字）
    paragraphs = [annual_report_text[i:i+512]
                  for i in range(0, min(len(annual_report_text), 5120), 512)]

    scores = {dim: 0.0 for dim in esr_dims}
    counts = {dim: 0 for dim in esr_dims}

    for para in paragraphs:
        if len(para.strip()) < 10:
            continue
        for dim, labels in esr_dims.items():
            result = classifier(para, candidate_labels=labels + ['其他'])
            # 取非"其他"类的最高得分
            for i, label in enumerate(result['labels']):
                if label != '其他':
                    scores[dim] += result['scores'][i]
                    counts[dim] += 1
                    break

    # 归一化得分到[0,100]
    final_scores = {}
    for dim in esr_dims:
        raw = scores[dim] / max(counts[dim], 1)
        final_scores[dim] = round(raw * 100, 2)

    final_scores['综合CSR得分'] = round(
        sum(final_scores.values()) / len(esr_dims), 2
    )
    return final_scores


# ============================================================
# 贡献四：Streamlit交互式排名系统（核心逻辑）
# 运行方式：streamlit run this_file.py
# ============================================================

# 单独保存为 ranking_app.py 后运行
STREAMLIT_APP_CODE = '''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="制造业上市公司价值500强", layout="wide")
st.title("🏭 制造业上市公司多维价值排名系统")

# 模拟数据（实际替换为真实数据库）
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "公司名称": [f"制造企业{i:03d}" for i in range(1, n+1)],
        "行业":     np.random.choice(["装备制造","汽车","电子","化工","钢铁"], n),
        "省份":     np.random.choice(["广东","浙江","江苏","山东","北京","上海"], n),
        "财务得分":  np.random.uniform(40, 100, n),
        "技术得分":  np.random.uniform(20, 100, n),
        "CSR得分":   np.random.uniform(30, 100, n),
        "供应链得分": np.random.uniform(20, 100, n),
    })

df = load_data()

# --- 侧边栏：权重调整 ---
st.sidebar.header("⚙️ 自定义权重")
w_fin  = st.sidebar.slider("财务权重",   0, 100, 40)
w_tech = st.sidebar.slider("技术权重",   0, 100, 30)
w_csr  = st.sidebar.slider("CSR权重",    0, 100, 20)
w_sup  = st.sidebar.slider("供应链权重", 0, 100, 10)
total  = w_fin + w_tech + w_csr + w_sup

st.sidebar.metric("权重合计", f"{total}（建议=100）")

# --- 筛选器 ---
col1, col2 = st.columns(2)
with col1:
    sel_industry = st.multiselect("按行业筛选", df["行业"].unique(), default=df["行业"].unique())
with col2:
    sel_province = st.multiselect("按省份筛选", df["省份"].unique(), default=df["省份"].unique())

filtered = df[df["行业"].isin(sel_industry) & df["省份"].isin(sel_province)].copy()

# --- 计算综合得分 ---
if total > 0:
    filtered["综合得分"] = (
        filtered["财务得分"]  * w_fin  / total +
        filtered["技术得分"]  * w_tech / total +
        filtered["CSR得分"]   * w_csr  / total +
        filtered["供应链得分"] * w_sup  / total
    ).round(2)
else:
    filtered["综合得分"] = 0

filtered = filtered.sort_values("综合得分", ascending=False).reset_index(drop=True)
filtered.index += 1

# --- 主内容区 ---
tab1, tab2, tab3 = st.tabs(["📋 排名榜单", "📊 分布分析", "🔍 企业详情"])

with tab1:
    st.subheader(f"Top 50 企业（共{len(filtered)}家）")
    st.dataframe(
        filtered[["公司名称","行业","省份","综合得分","财务得分","技术得分","CSR得分","供应链得分"]].head(50),
        use_container_width=True,
        height=500
    )
    csv = filtered.to_csv(index=True, encoding='utf-8-sig')
    st.download_button("📥 下载完整排名CSV", csv, "ranking.csv", "text/csv")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(filtered, x="综合得分", nbins=30,
                           title="综合得分分布", color_discrete_sequence=["steelblue"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.box(filtered, x="行业", y="综合得分",
                      title="各行业得分箱线图", color="行业")
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    company = st.selectbox("选择企业", filtered["公司名称"].head(50))
    row = filtered[filtered["公司名称"] == company].iloc[0]
    dims = ["财务得分", "技术得分", "CSR得分", "供应链得分"]
    vals = [row[d] for d in dims]
    fig3 = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=dims + [dims[0]],
        fill='toself', name=company,
        line_color='steelblue'
    ))
    fig3.update_layout(title=f"{company} 多维雷达图",
                       polar=dict(radialaxis=dict(range=[0,100])))
    st.plotly_chart(fig3, use_container_width=True)
    st.metric("综合排名", f"第 {filtered[filtered['公司名称']==company].index[0]} 名")
'''

def save_streamlit_app():
    with open("ranking_app.py", "w", encoding="utf-8") as f:
        f.write(STREAMLIT_APP_CODE)
    print("ranking_app.py 已保存，运行方式：streamlit run ranking_app.py")


# ============================================================
# 贡献五：内部培训 — 因果推断入门自动教学脚本
# 目标：生成一份HTML格式的交互式教学材料
# ============================================================

def generate_training_notebook():
    """生成因果推断培训用的Jupyter Notebook内容"""
    import json
    cells = [
        {
            "cell_type": "markdown",
            "source": ["# 因果推断入门：DID与合成控制法\n",
                       "## 适用于产业政策效果评估\n",
                       "**作者**：曹晨 | 机械工业经济管理研究院内训材料"]
        },
        {
            "cell_type": "markdown",
            "source": ["## 1. 核心问题：为什么需要因果推断？\n",
                       "- 简单前后对比的问题：没有政策会不会自然增长？\n",
                       "- 相关 ≠ 因果\n",
                       "- DID的核心思想：**找一个"如果没有政策会怎样"的参照组**"]
        },
        {
            "cell_type": "code",
            "source": ["# 演示：简单前后对比 vs DID的区别\n",
                       "import numpy as np, matplotlib.pyplot as plt\n",
                       "years = [2016,2017,2018,2019,2020]\n",
                       "treat = [100,102,110,115,120]  # 处理组\n",
                       "control= [100,103,106,109,112] # 对照组（反事实趋势）\n",
                       "# DID效应 = (处理组后-处理组前) - (对照组后-对照组前)\n",
                       "did_effect = (120-102) - (112-103)\n",
                       "print(f'简单前后对比：{120-102}（高估了！）')\n",
                       "print(f'DID估计真实效应：{did_effect}')"]
        }
    ]
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "cells": cells}
    with open("causal_inference_training.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print("培训Notebook已生成：causal_inference_training.ipynb")


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("【贡献二演示】产业政策效果评估")
    print("=" * 60)
    df = generate_panel_data()
    print(f"数据规模：{len(df)}条记录（{df['firm_id'].nunique()}家企业 × {df['year'].nunique()}年）")

    print("\n--- 双重差分估计 ---")
    did_result = run_did(df)

    print("\n--- 平行趋势检验 ---")
    pt_result = parallel_trend_test(df)

    print("\n--- 安慰剂检验 ---")
    placebo_test(df)

    print("\n--- 双重机器学习（DML）---")
    dml_est, te = run_dml(df)

    print("\n【贡献四】保存Streamlit排名App...")
    save_streamlit_app()

    print("\n【贡献五】生成培训Notebook...")
    generate_training_notebook()

    print("\n✅ 所有模块运行完毕")
    print("下一步：运行 `streamlit run ranking_app.py` 查看排名系统")