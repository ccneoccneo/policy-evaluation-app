"""
模块二：RAG知识库构建
功能：
  1. 将采集的政策文档切片（chunking）
  2. 用Embedding模型向量化
  3. 存入ChromaDB向量数据库
  4. 提供语义检索接口

安装依赖：
  pip install chromadb sentence-transformers langchain langchain-community
"""

import json
import re
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== 配置区 ==========
CHROMA_DB_PATH = "./chroma_policy_db"   # 向量数据库持久化路径
COLLECTION_NAME = "policy_research"     # 集合名称
CHUNK_SIZE = 500                        # 每个文本块字符数
CHUNK_OVERLAP = 50                      # 相邻块重叠字符数（保持上下文连贯）

# 使用中文友好的Embedding模型（本地运行，无需联网，保密合规）
# 备选模型：BAAI/bge-large-zh-v1.5（更准确，但较慢）
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"


# ========== 文本切片器 ==========
class TextChunker:
    """
    按段落+长度双重规则切片
    优先在句号/换行处断开，避免切断完整语义
    """
    def __init__(self, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        # 按自然段落先分割
        paragraphs = re.split(r'\n{2,}|(?<=[。！？])\n', text)
        chunks, buf = [], ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(buf) + len(para) <= self.chunk_size:
                buf += para + "\n"
            else:
                if buf:
                    chunks.append(buf.strip())
                # 若单段落超过chunk_size，强制按字符切
                if len(para) > self.chunk_size:
                    for i in range(0, len(para), self.chunk_size - self.overlap):
                        chunks.append(para[i:i + self.chunk_size])
                    buf = para[-(self.overlap):]
                else:
                    buf = para + "\n"

        if buf.strip():
            chunks.append(buf.strip())
        return [c for c in chunks if len(c) > 30]  # 过滤过短碎片


# ========== 向量知识库管理器 ==========
class PolicyVectorDB:
    def __init__(self):
        # 初始化ChromaDB（本地持久化存储）
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        # 使用本地Embedding模型（数据不出内网，满足保密要求）
        logger.info(f"加载Embedding模型：{EMBED_MODEL_NAME}（首次需下载）")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        # 自定义embedding函数接入ChromaDB
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL_NAME
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}  # 余弦相似度
        )
        self.chunker = TextChunker()
        logger.info(f"知识库就绪，当前文档块数：{self.collection.count()}")

    def add_documents(self, docs: List[Dict]):
        """批量添加文档到知识库"""
        ids, texts, metas = [], [], []

        for doc in docs:
            chunks = self.chunker.split(doc["content"])
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc['source']}_{hash(doc['url'])}_{i}"
                # 避免重复添加
                if chunk_id in ids:
                    continue
                ids.append(chunk_id)
                texts.append(chunk)
                metas.append({
                    "source": doc["source"],
                    "category": doc["category"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "crawl_time": doc.get("crawl_time", ""),
                    "chunk_index": i,
                })

        if not ids:
            return

        # 批量插入（每批100条，防止内存溢出）
        batch_size = 100
        for start in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[start:start+batch_size],
                documents=texts[start:start+batch_size],
                metadatas=metas[start:start+batch_size],
            )
        logger.info(f"成功入库 {len(ids)} 个文本块")

    def search(self, query: str, n_results=5, category_filter=None) -> List[Dict]:
        """
        语义检索：输入问题，返回最相关的文档片段
        支持按政策类别过滤（如只搜索"国资国企政策"）
        """
        where = {"category": category_filter} if category_filter else None
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        # 整理返回格式
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "content": doc,
                "title": meta["title"],
                "source": meta["source"],
                "url": meta["url"],
                "relevance_score": round(1 - dist, 4),  # 转为相关性分数（越高越相关）
            })
        return output

    def get_stats(self) -> Dict:
        """查看知识库统计信息"""
        count = self.collection.count()
        return {"total_chunks": count, "db_path": CHROMA_DB_PATH}


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 1. 初始化知识库
    db = PolicyVectorDB()

    # 2. 加载爬虫采集的文档（来自模块一）
    policy_data_dir = Path("policy_data")
    all_docs = []
    for f in policy_data_dir.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            all_docs.extend(json.load(fp))

    # 3. 也可以直接添加手动整理的文档
    manual_docs = [
        {
            "source": "国务院",
            "category": "宏观政策",
            "title": "关于全面振兴东北地区等老工业基地的若干意见",
            "url": "http://www.gov.cn/zhengce/...",
            "content": "此处粘贴政策全文...",
            "crawl_time": "2024-01-01",
        }
    ]
    all_docs.extend(manual_docs)

    # 4. 入库
    db.add_documents(all_docs)
    print(f"\n知识库统计：{db.get_stats()}")

    # 5. 测试检索
    query = "新能源汽车产业补贴政策对企业研发投入的影响"
    results = db.search(query, n_results=3)
    print(f"\n检索问题：{query}")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] 相关性：{r['relevance_score']}")
        print(f"    来源：{r['source']} - {r['title']}")
        print(f"    内容片段：{r['content'][:150]}...")