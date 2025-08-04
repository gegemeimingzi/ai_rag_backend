# models/llm_model.py
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件

def load_llm():
    """
    加载 Qwen 在线 API（OpenAI 兼容模式）
    """
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    if not api_key:
        raise ValueError("请设置环境变量 OPENAI_API_KEY 为你的阿里云 DashScope API Key")

    return ChatOpenAI(
        model="qwen-plus",   # 或 "qwen-plus"，"qwen-max" 等
        temperature=0.7,
        openai_api_key=api_key,
        openai_api_base=api_base
    )
