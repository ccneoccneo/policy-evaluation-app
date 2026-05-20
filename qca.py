"""
职业本科院校青年教师专业发展 NLP分析流水线
适配数据格式：国家级/省级/院校级政策文本 + 教师访谈文本
环境要求：pip install jieba transformers torch scikit-learn pandas numpy matplotlib
"""

import jieba
import jieba.analyse
import json
import re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# STEP 0：数据结构定义
# 说明：三级文本分层存储，便于后续分层比较分析
# ============================================================

# 国家级政策文本（示例，实际从教育部官网下载后填入）
national_texts = {
    "2019_职业技能提升行动方案": """
        加大职业培训力度，大规模开展职业技能培训，推动建立覆盖城乡全体劳动者、
        贯穿劳动者学习工作终身的职业培训制度。鼓励职业院校教师赴企业挂职锻炼，
        提升双师素质，建设高水平双师型教师队伍。
    """,
    "2020_职业教育提质培优行动计划": """
        完善职业学校教师资格标准，建立健全教师绩效评价机制。
        探索实施职称评审代表性成果制度，允许以专利成果、技术报告等替代论文要求。
        推行教师企业实践制度，每五年至少累计六个月在企业或实训基地实践。
    """,
    "2022_职业教育法": """
        职业学校应当建立健全教师培训制度，制定教师培训计划，
        保证教师参加培训的时间和条件。鼓励和支持专业技术人员、高技能人才
        到职业学校兼职任教。职业学校应当为教师参加企业实践提供条件保障。
    """
}

# 省级政策文本（示例）
provincial_texts = {
    "安徽_2021_职业教育改革实施方案": """
        推进双师型教师队伍建设，支持职业院校与企业共建双师型教师培养培训基地。
        改革职称评审制度，对取得职业技能等级证书的教师在职称评审中予以倾斜。
        设立职业教育发展专项资金，重点支持青年教师企业实训和能力提升。
    """,
    "广东_2021_职业教育创新发展实施方案": """
        建立健全教师分类评价机制，探索实施教学型、科研型、双师型教师分类评价。
        支持职业院校引进行业企业高技能人才担任专兼职教师。
        推进产教融合，建立稳定的产教融合型企业名单，推动企业深度参与教师培养。
    """
}

# 院校层面数据（你已有的格式）
school_data = {
    "芜湖职业技术学院": {
        "province": "安徽",
        "type": "综合",
        "policy_text": """
        实行劳模工匠进校园项目，邀请企业劳模担任兼职教师，
        与青年教师结对培养。设立脱产实训制度，青年教师可申请
        三到六个月脱产到企业参加实训。建立横向课题激励机制，
        横向经费提成比例高于纵向课题。教师发展中心提供个性化
        发展咨询服务。职称评审设立双师型专项通道，取得职业技能
        等级证书可替代部分论文要求。设立青年教师创新基金，
        每年资助优秀项目十项。
        """,
        "interview_text": """
        脱产实训的机会很难得，我去了一家制造企业待了四个月，
        学到了很多实际的东西，回来之后课讲得更有底气了。
        横向课题的激励也不错，让我主动去联系企业合作。
        但有时候觉得自己的发展方向不是特别清晰，学校的整体
        规划感不如一些大院校强，需要自己摸索的地方比较多。
        """
    },
    "南京工业职业技术大学": {
        "province": "江苏",
        "type": "工科",
        "policy_text": """
        实行五类教师岗位分类评价体系，包括教学型、科研型、双师型、
        管理型和社会服务型。设立职称评审绿色通道，产教融合成果突出者
        可低职高聘。博士入企双轨制培养，新入职博士须完成企业项目实践。
        设立企业教师工作站，鼓励教师深度嵌入企业技术研发。
        企业实践纳入职称评审刚性约束，须完成不少于六个月的企业实践。
        """,
        "interview_text": """
        学校给了我们很多选择，我可以走教学型也可以走双师型，
        这让我能根据自己的特长来规划发展路径。
        企业实践确实有点硬性要求，但去了之后发现真的很有用，
        现在横向项目也接了好几个。学校在科研起步阶段给了缓冲期，
        压力没那么大，可以慢慢摸索方向。
        """
    },
    "深圳职业技术大学": {
        "province": "广东",
        "type": "综合",
        "policy_text": """
        实施十年制教师发展长期规划，分阶段设立发展目标。
        建立代表作评价制度，论文、专利、作品、技术报告均可作为
        职称评审代表作。设立专项科研基金，为青年教师提供启动经费。
        与香港理工大学共建博士联培项目。建立十六个特色产业学院，
        教师嵌入产业学院开展双师培育。破五唯评价体系，
        重大产业贡献可破格晋升。
        """,
        "interview_text": """
        代表作制度对我帮助很大，我的技术成果可以直接用于职称评审，
        不用像以前那样非得发论文。产业学院的平台也很好，
        和企业的合作很自然就建立起来了。学校给了我们充分的空间
        去探索自己的方向，十年规划让我知道自己在哪个阶段应该做什么。
        """
    }
}

# ============================================================
# STEP 1：文本预处理
# 说明：构建职教专域停用词表 + 自定义词典，这是NLP质量的基础
# ============================================================

# 程式化行政套话停用词（需过滤，否则干扰主题提取）
admin_stopwords = [
    "根据", "为了", "特制定", "本办法", "依据", "按照", "有关", "相关",
    "工作", "进行", "开展", "加强", "推进", "建立", "完善", "制度",
    "学校", "院校", "教师", "发展", "管理", "实施", "落实", "要求"
]

# 职教专域关键词典（帮助jieba正确识别专业词汇）
vocational_edu_dict = [
    "双师型", "产教融合", "职业本科", "代表作评价", "低职高聘",
    "横向课题", "纵向课题", "脱产实训", "企业工作站", "产业学院",
    "破五唯", "绿色通道", "分类评价", "缓冲期", "揭榜挂帅",
    "职业技能等级证书", "青年教师", "劳模工匠", "科研启动经费"
]

# 能动性词典（三类：主动型/遵从型/策略型）
agency_dict = {
    "active": [  # 主动内化型：体现真实能动意愿
        "主动", "自主", "选择", "规划", "探索", "尝试", "突破",
        "喜欢", "感兴趣", "有底气", "愿意", "自己决定"
    ],
    "compliant": [  # 被动遵从型：体现制度压力下的服从
        "不得不", "被要求", "硬性", "没有选择", "必须", "压力",
        "只能", "要求我们", "规定"
    ],
    "strategic": [  # 策略套利型：用制度语言包装自身利益
        "利用", "借助", "用这个来", "可以替代", "算作",
        "包装", "对应", "换算"
    ]
}


def preprocess_text(text, custom_dict=None, stopwords=None):
    """文本预处理：清洗 → 分词 → 过滤"""
    # 加载自定义词典
    if custom_dict:
        for word in custom_dict:
            jieba.add_word(word)

    # 基础清洗
    text = re.sub(r'\s+', '', text)  # 去空白
    text = re.sub(r'[，。；：""''【】（）]', ' ', text)  # 标点转空格
    text = re.sub(r'\d+', 'NUM', text)  # 数字统一为NUM

    # 分词
    words = jieba.lcut(text)

    # 过滤：去停用词 + 去单字（单字在政策文本中通常无语义）
    if stopwords:
        words = [w for w in words if w not in stopwords and len(w) > 1]
    else:
        words = [w for w in words if len(w) > 1]

    return words


# 对所有文本预处理
print("=== STEP 1: 文本预处理 ===")
processed = {}

for level, texts in [("national", national_texts), ("provincial", provincial_texts)]:
    processed[level] = {}
    for doc_id, text in texts.items():
        words = preprocess_text(text, vocational_edu_dict, admin_stopwords)
        processed[level][doc_id] = words
        print(f"[{level}] {doc_id}: {len(words)}个有效词")

processed["school"] = {}
for school, data in school_data.items():
    for text_type in ["policy_text", "interview_text"]:
        key = f"{school}_{text_type}"
        words = preprocess_text(data[text_type], vocational_edu_dict, admin_stopwords)
        processed["school"][key] = words
        print(f"[school] {school} - {text_type}: {len(words)}个有效词")

# ============================================================
# STEP 2：TF-IDF分析 → 测量C1（政策资源密度）
# 说明：计算每所院校政策文本中双师/产教/评价改革词汇的相对权重
# ============================================================

print("\n=== STEP 2: TF-IDF分析（政策资源密度） ===")

# 核心议题词典（对应6个条件变量）
topic_keywords = {
    "C1_政策密度": ["制度", "办法", "规定", "标准", "方案", "通知"],
    "C2_组织载体": ["发展中心", "产业学院", "工作站", "平台", "基地", "机构"],
    "C3_资源投入": ["经费", "资助", "基金", "名额", "项目", "支持"],
    "C4_评价多元": ["代表作", "分类", "双师", "替代", "多元", "专项通道"],
    "C5_试错容忍": ["缓冲", "容错", "破格", "试用", "弹性", "灵活"],
    "C6_自主选择": ["自主", "选择", "个性化", "规划", "可申请", "自愿"]
}


def compute_topic_density(text_words, topic_keywords):
    """计算文本在各条件变量维度上的关键词密度"""
    word_freq = Counter(text_words)
    total = len(text_words) if text_words else 1
    density = {}
    for topic, keywords in topic_keywords.items():
        count = sum(word_freq.get(kw, 0) for kw in keywords)
        density[topic] = round(count / total * 100, 3)  # 每百词密度
    return density


# 计算各院校政策文本的主题密度
school_density = {}
for school in school_data:
    key = f"{school}_policy_text"
    words = processed["school"].get(key, [])
    density = compute_topic_density(words, topic_keywords)
    school_density[school] = density
    print(f"\n{school} 条件变量密度:")
    for k, v in density.items():
        bar = "█" * int(v * 10)
        print(f"  {k}: {v:.3f} {bar}")

# ============================================================
# STEP 3：LDA主题模型 → 发现隐含主题结构
# 说明：无监督识别文本主题，验证变量框架是否有数据层面支撑
# ============================================================

print("\n=== STEP 3: LDA主题模型 ===")

# 合并所有院校政策文本
all_policy_texts = []
school_names = []
for school, data in school_data.items():
    words = preprocess_text(data["policy_text"], vocational_edu_dict, admin_stopwords)
    all_policy_texts.append(" ".join(words))
    school_names.append(school)

# 也加入国家和省级文本
for doc_id, text in national_texts.items():
    words = preprocess_text(text, vocational_edu_dict, admin_stopwords)
    all_policy_texts.append(" ".join(words))
    school_names.append(f"国家_{doc_id[:10]}")

# TF-IDF向量化
vectorizer = TfidfVectorizer(max_features=200, min_df=1)
tfidf_matrix = vectorizer.fit_transform(all_policy_texts)
feature_names = vectorizer.get_feature_names_out()

# LDA主题模型（设K=6，对应6个条件变量维度）
lda = LatentDirichletAllocation(
    n_components=6,
    random_state=42,
    max_iter=50
)
lda.fit(tfidf_matrix)

# 输出每个主题的top词汇
print("LDA识别的6个主题（应与条件变量维度对应）:")
topic_labels = ["主题A", "主题B", "主题C", "主题D", "主题E", "主题F"]
for i, (topic, label) in enumerate(zip(lda.components_, topic_labels)):
    top_words = [feature_names[j] for j in topic.argsort()[-8:][::-1]]
    print(f"  {label}: {' | '.join(top_words)}")

# 各文档的主题分布
doc_topic_dist = lda.transform(tfidf_matrix)
print("\n各院校/层级文本的主题分布（行=文档，列=主题占比）:")
df_topics = pd.DataFrame(
    doc_topic_dist,
    index=school_names,
    columns=[f"主题{i + 1}" for i in range(6)]
).round(3)
print(df_topics.to_string())

# ============================================================
# STEP 4：能动性分析 → 测量C4/C5/C6（能动空间三变量）
# 说明：对访谈文本进行三类能动立场识别
# ============================================================

print("\n=== STEP 4: 教师能动性分析 ===")


def analyze_agency(interview_text, agency_dict):
    """
    分析访谈文本中的能动性立场分布
    返回：主动型/遵从型/策略型比例 + 能动空间综合得分
    """
    words = preprocess_text(interview_text, vocational_edu_dict)
    word_set = set(words)

    counts = {}
    for agency_type, keywords in agency_dict.items():
        # 同时检查分词结果和原始文本（处理多字词汇）
        count = sum(1 for kw in keywords if kw in interview_text)
        counts[agency_type] = count

    total = sum(counts.values()) or 1
    ratios = {k: round(v / total, 3) for k, v in counts.items()}

    # 能动空间综合得分（主动型权重最高，策略型次之，遵从型负向）
    agency_score = (
            ratios["active"] * 1.0 +
            ratios["strategic"] * 0.3 -
            ratios["compliant"] * 0.8
    )
    agency_score = round(max(0, min(1, agency_score + 0.5)), 3)  # 归一化到0-1

    return ratios, agency_score, counts


# 对各院校访谈文本进行能动性分析
print(f"{'院校':<15} {'主动型':>8} {'遵从型':>8} {'策略型':>8} {'能动得分':>10}")
print("-" * 55)

agency_results = {}
for school, data in school_data.items():
    ratios, score, counts = analyze_agency(data["interview_text"], agency_dict)
    agency_results[school] = {"ratios": ratios, "score": score}
    print(f"{school:<15} {ratios['active']:>8.3f} {ratios['compliant']:>8.3f} "
          f"{ratios['strategic']:>8.3f} {score:>10.3f}")

# ============================================================
# STEP 5：语义相似度分析 → 测量制度传导强度
# 说明：计算国家政策→省级政策→院校政策的语义继承程度
# 这直接反映"制度厚度"中的制度一致性维度
# ============================================================

print("\n=== STEP 5: 三级制度传导强度分析 ===")


def compute_transmission_strength(national_texts, provincial_texts, school_data, vectorizer):
    """
    计算国家→省→院校三级政策的语义相似度
    相似度高 = 院校政策忠实执行上级精神（制度传导强）
    相似度低 = 院校政策有较大本土创新（能动空间大）
    """
    # 向量化各级文本
    nat_text = " ".join([
        " ".join(preprocess_text(t, vocational_edu_dict, admin_stopwords))
        for t in national_texts.values()
    ])

    prov_texts_by_province = defaultdict(list)
    for doc_id, text in provincial_texts.items():
        province = doc_id.split("_")[0]
        words = preprocess_text(text, vocational_edu_dict, admin_stopwords)
        prov_texts_by_province[province].append(" ".join(words))

    results = {}
    for school, data in school_data.items():
        province = data["province"]
        school_words = preprocess_text(
            data["policy_text"], vocational_edu_dict, admin_stopwords
        )
        school_text = " ".join(school_words)

        # 获取对应省份文本
        prov_text = " ".join(prov_texts_by_province.get(province, [nat_text]))

        # 计算相似度（需要重新fit包含这些文本的vectorizer）
        texts_to_compare = [nat_text, prov_text, school_text]
        local_vec = TfidfVectorizer(max_features=100, min_df=1)
        try:
            mat = local_vec.fit_transform(texts_to_compare)
            nat_school_sim = cosine_similarity(mat[0:1], mat[2:3])[0][0]
            prov_school_sim = cosine_similarity(mat[1:2], mat[2:3])[0][0]
        except:
            nat_school_sim = prov_school_sim = 0.0

        results[school] = {
            "国家-院校相似度": round(nat_school_sim, 3),
            "省级-院校相似度": round(prov_school_sim, 3),
            # 创新空间 = 1 - 平均传导相似度（相似度低说明院校有更多本土创新）
            "本土创新空间": round(1 - (nat_school_sim + prov_school_sim) / 2, 3)
        }

    return results


transmission = compute_transmission_strength(
    national_texts, provincial_texts, school_data, vectorizer
)

print(f"{'院校':<15} {'国家-院校':>10} {'省级-院校':>10} {'本土创新空间':>12}")
print("-" * 55)
for school, metrics in transmission.items():
    print(f"{school:<15} {metrics['国家-院校相似度']:>10.3f} "
          f"{metrics['省级-院校相似度']:>10.3f} "
          f"{metrics['本土创新空间']:>12.3f}")

# ============================================================
# STEP 6：综合输出 → 为fsQCA校准提供数据基础
# 说明：整合以上分析，生成每所院校在6个条件变量上的原始得分
# 这些得分将作为fsQCA校准的客观依据（不直接是模糊集分数）
# ============================================================

print("\n=== STEP 6: fsQCA校准参考数据汇总 ===")

fsqca_data = []
for school in school_data:
    row = {"院校": school, "省份": school_data[school]["province"]}

    # 制度厚度三变量（来自TF-IDF密度分析）
    density = school_density.get(school, {})
    row["C1_政策资源密度_原始分"] = density.get("C1_政策密度", 0)
    row["C2_组织载体_原始分"] = density.get("C2_组织载体", 0)
    row["C3_资源投入_原始分"] = density.get("C3_资源投入", 0)

    # 能动空间三变量（来自能动性分析）
    agency = agency_results.get(school, {})
    ratios = agency.get("ratios", {})
    row["C4_评价多元_原始分"] = density.get("C4_评价多元", 0)
    row["C5_试错容忍_原始分"] = density.get("C5_试错容忍", 0)
    row["C6_自主选择_原始分"] = agency.get("score", 0)

    # 传导强度（辅助参考）
    trans = transmission.get(school, {})
    row["制度传导强度"] = 1 - trans.get("本土创新空间", 0)
    row["本土创新空间"] = trans.get("本土创新空间", 0)

    fsqca_data.append(row)

df_fsqca = pd.DataFrame(fsqca_data)
print(df_fsqca.to_string(index=False))

print("""
=== 下一步操作说明 ===

1. 将上表中C1-C6的原始分数导出为Excel
2. 在fsQCA 3.0软件中，根据三个锚点进行校准：
   - 完全隶属(1.0)锚点：根据理论判断设为某个值（如得分>0.8的院校特征）
   - 交叉点(0.5)锚点：样本中位数附近
   - 完全不隶属(0.0)锚点：得分最低的院校特征
3. NLP输出的原始分只是校准依据之一，最终校准需结合：
   - 制度文本的人工深度阅读
   - 访谈资料的质性判断
   - 公开年报等第三方数据
""")