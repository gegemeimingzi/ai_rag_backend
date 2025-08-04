import os
from langchain_redis import RedisChatMessageHistory
from dotenv import load_dotenv

load_dotenv()
_redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

# 如果你想单例管理，可以：
_redis_histories = {}

def get_message_history(session_id):
    if session_id not in _redis_histories:
        _redis_histories[session_id] = RedisChatMessageHistory(session_id=session_id, redis_url=_redis_url)
    return _redis_histories[session_id]