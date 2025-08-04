import re
from typing import List, Tuple
from langchain.chains import LLMChain
from prompts.rerank_prompt import rerank_prompt


def get_rerank_scores(
    llm,
    question: str,
    documents: List[str],
    default_score: float = 0.0,
    default_reason: str = "未能识别评分格式",
    verbose: bool = False
) -> List[Tuple[float, str, str]]:
    """
    使用 LLM 对候选文档进行相关性打分，并返回每个文档的评分、原文、理由。

    Args:
        llm: LLM 实例（如 ChatOpenAI、Qwen、glm 等）
        question (str): 用户问题
        documents (List[str]): 候选文本列表
        default_score (float): 未匹配时默认分数
        default_reason (str): 未匹配时默认理由
        verbose (bool): 是否打印每条原始响应（调试用）

    Returns:
        List[Tuple[score: float, doc: str, reason: str]]
    """
    chain = LLMChain(llm=llm, prompt=rerank_prompt)
    results = []

    for idx, doc in enumerate(documents):
        try:
            response = chain.run({"question": question, "context": doc})
            if verbose:
                print(f"[LLM 原始响应 #{idx + 1}]:\n{response.strip()}\n")

            # 使用强健正则提取“评分: <数字>”“理由: <文本>”
            match = re.search(
                r"评分[:：]?\s*(\d+(?:\.\d+)?)\s*[\r\n]+理由[:：]?\s*(.+)",
                response,
                re.IGNORECASE | re.DOTALL
            )

            if match:
                score = float(match.group(1))
                reason = match.group(2).strip()
                results.append((score, doc, reason))
            else:
                results.append((default_score, doc, default_reason))

        except Exception as e:
            # 如果模型调用失败或出错，记录并继续
            if verbose:
                print(f"[重排序异常] 文档 #{idx + 1}: {str(e)}")
            results.append((default_score, doc, "模型调用异常"))

    return results
