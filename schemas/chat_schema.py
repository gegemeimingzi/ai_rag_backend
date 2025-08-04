from pydantic import BaseModel
from typing import List, Optional


class Message(BaseModel):
    """
    聊天消息模型，表示用户或助理发送的单条消息
    """
    role: str  # "user" 或 "assistant"
    content: str


class SourceDocument(BaseModel):
    """
    检索到的知识文档结构，包含内容、评分、理由及法律依据来源信息
    """
    content: str
    score: Optional[float] = None
    reason: Optional[str] = None
    source_name: Optional[str] = None  # 文件简短名（文件名不含后缀）
    source_text: Optional[str] = None  # 法条条文标题


class ChatRequest(BaseModel):
    """
    聊天请求体，包含会话ID和消息列表
    """
    session_id: str
    messages: List[Message]


class ChatResponse(BaseModel):
    """
    聊天接口响应体，包含回答、是否用到知识库、相似度评分、原因及相关文档信息
    """
    answer: str
    used_knowledge: bool
    top_score: Optional[float] = None
    no_use_reason: Optional[str] = None
    source_documents: Optional[List[SourceDocument]] = None
