from langchain_core.messages import HumanMessage, AIMessage
from typing import List
from schemas.chat_schema import ChatRequest, ChatResponse, Message, SourceDocument
from memory.redis_memory import get_message_history
from rag_chain.chain import get_qa_chain
from models.llm_model import load_llm
from models.embed_model import load_embedding_model
import tiktoken

# 配置项
MAX_HISTORY_MSGS = 10
MAX_TOKENS = 1500
ENCODING_NAME = "gpt2"

# 加载模型与问答链（可在外层调用初始化后传入优化性能）
llm = load_llm()
embedding = load_embedding_model()
qa_chain = get_qa_chain(llm, embedding)


def count_tokens(text: str, encoding_name=ENCODING_NAME) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def build_history_text_with_token_limit(messages: List, max_msgs=MAX_HISTORY_MSGS, max_tokens=MAX_TOKENS) -> str:
    history_text = ""
    total_tokens = 0

    for msg in reversed(messages[-max_msgs:]):
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        line = f"{role}: {msg.content}\n"
        line_tokens = count_tokens(line)

        if total_tokens + line_tokens > max_tokens:
            break

        history_text = line + history_text
        total_tokens += line_tokens

    return history_text


def chat_service(req: ChatRequest) -> ChatResponse:
    history = get_message_history(req.session_id)
    last_user_question = None

    # 写入历史消息
    for msg in req.messages:
        if msg.role == "user":
            history.add_message(HumanMessage(content=msg.content))
            last_user_question = msg.content
        elif msg.role == "assistant":
            history.add_message(AIMessage(content=msg.content))

    if not last_user_question:
        raise ValueError("No user question provided.")

    # 构建上下文历史
    history_text = build_history_text_with_token_limit(history.messages)

    # 调用 RAG 链
    result = qa_chain.invoke({
        "query": last_user_question,
        "chat_history": history_text
    })

    answer = result.get("result", "")
    used_knowledge = result.get("used_knowledge", False)
    top_score = result.get("top_score", None)
    no_use_reason = result.get("no_use_reason", None)
    raw_docs = result.get("source_documents", [])

    # 构造文档响应结构
    source_documents = []
    for item in raw_docs:
        if isinstance(item, dict):
            source_documents.append(SourceDocument(**item))
        elif isinstance(item, str):
            source_documents.append(SourceDocument(content=item))
        else:
            try:
                score, content, reason_text, source_name, source_text = item
                source_documents.append(SourceDocument(
                    content=content,
                    score=score,
                    reason=reason_text,
                    source_name=source_name,
                    source_text=source_text
                ))
            except Exception:
                pass

    # 追加 AI 回复进历史
    history.add_message(AIMessage(content=answer))

    return ChatResponse(
        answer=answer,
        used_knowledge=used_knowledge,
        top_score=top_score,
        no_use_reason=no_use_reason,
        source_documents=source_documents
    )
