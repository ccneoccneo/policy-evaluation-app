"""
模块三：LLM问答引擎（RAG核心）
功能：
  1. 接收研究员的问题
  2. 从知识库检索相关政策文档片段
  3. 将检索结果+问题组合成Prompt，调用LLM生成回答
  4. 支持本地模型（保密合规）和API两种模式

安装依赖：
  pip install openai transformers torch accelerate
  本地模型推荐：Qwen2.5-7B-Instruct 或 DeepSeek-R1-Distill-Qwen-7B
"""

from typing import List, Dict, Optional
import json
import logging
from module2_vectordb import PolicyVectorDB  # 引入模块二

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== Prompt模板 ==========
SYSTEM_PROMPT = """你是机械工业经济管理研究院的专业政策研究助手。
你的职责是：根据提供的政策文件原文，准确、专业地回答研究员的问题。

回答要求：
1. 严格基于提供的参考资料回答，不要编造内容
2. 使用政策研究报告的专业语言风格
3. 回答要结构清晰，适当分点阐述
4. 在回答末尾标注主要参考的文件来源
5. 如果参考资料不足以回答问题，请明确说明需要补充哪些资料
"""

RAG_PROMPT_TEMPLATE = """
【参考政策文件】
{context}

【研究员问题】
{question}

【回答要求】
请基于上述政策文件，给出专业、准确的分析回答。
"""


# ========== LLM调用层（支持两种模式） ==========
class LLMBackend:
    """
    模式一：本地部署（推荐，数据不出内网）
    模式二：API调用（开发测试用，生产环境慎用）
    """

    def __init__(self, mode="local", model_path=None, api_key=None, api_base=None):
        self.mode = mode
        if mode == "local":
            self._init_local(model_path)
        else:
            self._init_api(api_key, api_base)

    def _init_local(self, model_path: str):
        """加载本地模型（Qwen/DeepSeek等国产模型）"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        logger.info(f"加载本地模型：{model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",  # 自动分配GPU/CPU
            trust_remote_code=True,
        )
        logger.info("本地模型加载完成")

    def _init_api(self, api_key: str, api_base: str):
        """API模式（兼容OpenAI格式，支持私有化部署的vLLM服务）"""
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        logger.info(f"API模式初始化完成：{api_base}")

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens=2048, temperature=0.3) -> str:
        if self.mode == "local":
            return self._generate_local(system_prompt, user_prompt, max_tokens, temperature)
        else:
            return self._generate_api(system_prompt, user_prompt, max_tokens, temperature)

    def _generate_local(self, sys_p, user_p, max_tokens, temperature) -> str:
        messages = [{"role": "system", "content": sys_p},
                    {"role": "user", "content": user_p}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=temperature,
                do_sample=temperature > 0, pad_token_id=self.tokenizer.eos_token_id
            )
        generated = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _generate_api(self, sys_p, user_p, max_tokens, temperature) -> str:
        resp = self.client.chat.completions.create(
            model="qwen2.5-7b-instruct",  # 替换为实际部署的模型名
            messages=[{"role": "system", "content": sys_p},
                      {"role": "user", "content": user_p}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content


# ========== RAG问答引擎 ==========
class PolicyRAGEngine:
    def __init__(self, llm: LLMBackend, db: PolicyVectorDB,
                 n_retrieve=5, relevance_threshold=0.5):
        self.llm = llm
        self.db = db
        self.n_retrieve = n_retrieve
        self.threshold = relevance_threshold
        self.history: List[Dict] = []  # 多轮对话历史

    def _build_context(self, retrieved: List[Dict]) -> str:
        """将检索结果格式化为Prompt上下文"""
        ctx_parts = []
        for i, r in enumerate(retrieved, 1):
            ctx_parts.append(
                f"【文件{i}】来源：{r['source']} | 标题：{r['title']}\n"
                f"相关度：{r['relevance_score']}\n"
                f"内容：{r['content']}\n"
            )
        return "\n---\n".join(ctx_parts)

    def query(self, question: str, category_filter: Optional[str] = None,
              verbose=True) -> Dict:
        """
        核心问答接口
        Args:
            question: 研究员的问题
            category_filter: 可选，按政策类别过滤（"宏观政策"/"工业政策"/"国资国企政策"）
            verbose: 是否打印检索详情
        Returns:
            包含答案、来源引用的字典
        """
        # Step 1: 语义检索相关文档
        retrieved = self.db.search(question, self.n_retrieve, category_filter)
        # 过滤低相关度结果
        retrieved = [r for r in retrieved if r["relevance_score"] >= self.threshold]

        if not retrieved:
            return {"answer": "未找到足够相关的政策文件，建议补充相关资料后再提问。",
                    "sources": [], "retrieved_count": 0}

        if verbose:
            logger.info(f"检索到 {len(retrieved)} 条相关文档片段")

        # Step 2: 构建Prompt
        context = self._build_context(retrieved)
        user_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        # Step 3: 调用LLM生成回答
        answer = self.llm.generate(SYSTEM_PROMPT, user_prompt)

        # Step 4: 整理来源引用
        sources = [{"title": r["title"], "source": r["source"],
                    "url": r["url"], "score": r["relevance_score"]}
                   for r in retrieved]

        # 记录对话历史
        self.history.append({"question": question, "answer": answer})

        return {"answer": answer, "sources": sources, "retrieved_count": len(retrieved)}

    def batch_query(self, questions: List[str]) -> List[Dict]:
        """批量处理多个问题（用于课题调研阶段批量分析政策）"""
        return [self.query(q, verbose=False) for q in questions]

    def generate_policy_summary(self, topic: str) -> str:
        """
        专项功能：生成特定主题的政策综述
        用途：快速生成报告中"政策背景"章节的初稿
        """
        question = f"""请对"{topic}"相关政策进行系统梳理，按以下结构输出：
        1. 政策演变脉络（时间线）
        2. 核心政策目标
        3. 主要政策工具
        4. 重点支持方向
        5. 政策空白与不足
        """
        result = self.query(question)
        return result["answer"]

    def generate_report_section(self, section_title: str, key_points: List[str]) -> str:
        """
        专项功能：基于知识库生成报告章节初稿
        用途：加速报告写作，生成有政策依据的文字
        """
        points_str = "\n".join(f"- {p}" for p in key_points)
        question = f"""请基于相关政策文件，为研究报告的"{section_title}"章节撰写初稿。
        需要覆盖以下要点：
        {points_str}

        要求：语言风格符合政府研究报告规范，逻辑严密，有理有据。
        """
        result = self.query(question)
        return result["answer"]


# ========== 使用示例 ==========
if __name__ == "__main__":
    # ---- 初始化知识库 ----
    db = PolicyVectorDB()

    # ---- 初始化LLM（二选一）----

    # 方案A：本地部署（生产环境推荐，数据安全）
    # llm = LLMBackend(mode="local", model_path="/models/Qwen2.5-7B-Instruct")

    # 方案B：API模式（开发测试，或内网vLLM服务）
    llm = LLMBackend(
        mode="api",
        api_key="your-key",
        api_base="http://localhost:8000/v1"  # 内网vLLM地址
    )

    # ---- 初始化RAG引擎 ----
    engine = PolicyRAGEngine(llm, db, n_retrieve=5, relevance_threshold=0.45)

    # ---- 场景1：直接问答 ----
    q1 = "十四五期间国家对装备制造业的主要支持政策有哪些？"
    result = engine.query(q1, category_filter="工业政策")
    print("=" * 60)
    print(f"问题：{q1}")
    print(f"\n回答：\n{result['answer']}")
    print(f"\n引用来源（{len(result['sources'])}条）：")
    for s in result["sources"]:
        print(f"  [{s['score']}] {s['source']} - {s['title']}")

    # ---- 场景2：生成政策背景综述（用于报告写作）----
    print("\n" + "=" * 60)
    summary = engine.generate_policy_summary("新能源汽车产业发展")
    print(f"政策综述：\n{summary}")

    # ---- 场景3：生成报告章节 ----
    print("\n" + "=" * 60)
    section = engine.generate_report_section(
        section_title="我国装备制造业政策支持现状分析",
        key_points=["财税支持政策", "技术创新政策", "国产替代推进政策", "重大装备首台套政策"]
    )
    print(f"报告章节初稿：\n{section}")