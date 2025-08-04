import logging
import jieba
from langchain.chains.llm import LLMChain
from langchain_core.runnables import Runnable
from langchain.schema import Document
from langchain_community.vectorstores import Chroma  # 这里根据新版改用 langchain_community
from langchain_community.retrievers import BM25Retriever

from rerank.rerank_chain import get_rerank_scores
from prompts.legal_prompt import legal_prompt
from prompts.fallback_prompt import fallback_prompt

logger = logging.getLogger(__name__)


class QARunnable(Runnable):
    """
    中文法律问答链（混合检索版）：
    支持向量检索 + BM25 关键词检索（基于中文分词）+ LLM重排序 + 上下文拼接 + 大模型问答。
    """

    def __init__(self, llm, embedding, persist_dir: str = "chroma_persist", threshold: float = 70):
        """
        初始化问答链组件。

        Args:
            llm: 大语言模型实例（如 Qwen/Qianwen API）
            embedding: 嵌入模型实例（用于 Chroma 向量库）
            persist_dir: Chroma 向量数据库持久化目录
            threshold: LLM 重排序保留阈值，低于该值的文档将被过滤
        """
        # 初始化 Chroma 向量数据库
        self.vectordb = Chroma(
            embedding_function=embedding,
            persist_directory=persist_dir,
            collection_name="legal_docs"
        )

        # 从向量库获取所有文档内容和元数据（对应列表）
        vectordb_data = self.vectordb.get()
        texts = vectordb_data.get('documents', [])  # 文本列表，类型为 List[str]
        metadatas = vectordb_data.get('metadatas', [])  # 元数据列表，类型为 List[dict]

        # 对文档内容进行中文分词，构建分词后的 Document 列表，供 BM25 使用
        segmented_docs = []
        for text, meta in zip(texts, metadatas):
            tokens = list(jieba.cut(text))  # 中文分词
            segmented_text = " ".join(tokens)  # 分词结果用空格连接
            segmented_docs.append(Document(page_content=segmented_text, metadata=meta))

        # 初始化 BM25 检索器，基于分词后的文档
        self.bm25 = BM25Retriever.from_documents(segmented_docs)

        # 初始化两个 LLMChain，分别处理有上下文和无上下文的回答
        self.chain_with_context = LLMChain(llm=llm, prompt=legal_prompt)
        self.fallback_prompt = LLMChain(llm=llm, prompt=fallback_prompt)

        self.threshold = threshold
        self.max_context_length = 2000  # 限制上下文字符数，防止 prompt 过长报错

    def invoke(self, inputs: dict) -> dict:
        """
        执行问答流程。

        Args:
            inputs: 包含用户 query 和可选 chat_history 的字典

        Returns:
            dict: 包括答案、所用文档、得分、是否使用知识、原因等
        """
        query = inputs.get("query", "")
        chat_history = inputs.get("chat_history", "")
        logger.info(f"Received query: {query}")

        # Step 1: 向量检索（Top 10）
        vector_docs = self.vectordb.similarity_search(query, k=15)

        # Step 2: BM25 检索（Top 10，基于中文分词）
        bm25_docs = self.bm25.invoke(query)[:5]

        # Step 3: 合并检索结果并去重
        seen = set()
        merged_docs = []
        for doc in vector_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged_docs.append(doc)

        if not merged_docs:
            logger.info("No documents retrieved from hybrid search.")
            result = self.fallback_prompt.run({"question": query})
            return {
                "result": result,
                "source_documents": [],
                "top_score": 0.0,
                "used_knowledge": False,
                "no_use_reason": "未检索到任何文档"
            }

        # Step 4: 用 LLM 进行重排序
        raw_docs = [doc.page_content for doc in merged_docs]
        reranked = get_rerank_scores(self.chain_with_context.llm, query, raw_docs)

        # 过滤低于阈值的文档，按得分降序排序
        filtered = sorted(
            [item for item in reranked if item[0] >= self.threshold],
            key=lambda x: x[0],
            reverse=True
        )

        filtered = filtered[:3]

        if not filtered:
            logger.info(f"No documents passed the relevance threshold (≥ {self.threshold}).")
            result = self.fallback_prompt.run({"question": query, "chat_history": chat_history})
            return {
                "result": result,
                "source_documents": [],
                "top_score": 0.0,
                "used_knowledge": False,
                "no_use_reason": "无相关知识文档或该问题不需要引用文档"
            }

        # Step 5: 拼接上下文，记录文档元信息
        content2metadata = {doc.page_content: doc.metadata for doc in merged_docs}

        context = ""
        selected_docs = []
        for score, doc_text, reason in filtered:
            if len(context) + len(doc_text) > self.max_context_length:
                break

            metadata = content2metadata.get(doc_text)
            source_name = metadata.get("source_name", "") if metadata else ""
            source_text = metadata.get("source_text", "") if metadata else ""

            context += doc_text + "\n\n"
            selected_docs.append({
                "content": doc_text,
                "score": score,
                "reason": reason,
                "source_name": source_name,
                "source_text": source_text
            })

        top_score = selected_docs[0]["score"] if selected_docs else 0.0
        logger.info(f"Using reranked context with top score: {top_score}")

        # Step 6: 调用大模型生成回答
        response = self.chain_with_context.run({
            "context": context,
            "question": query,
            "chat_history": chat_history
        })

        return {
            "result": response,
            "source_documents": selected_docs,
            "top_score": top_score,
            "used_knowledge": True
        }


def get_qa_chain(llm, embedding, persist_dir: str = "chroma_persist", threshold: float = 70) -> QARunnable:
    """
    构建并返回问答链实例。

    Args:
        llm: 大语言模型
        embedding: 嵌入模型
        persist_dir: 向量存储目录
        threshold: LLM 重排序阈值

    Returns:
        QARunnable: 中文法律问答链
    """
    return QARunnable(llm, embedding, persist_dir, threshold)
