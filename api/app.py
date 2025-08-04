from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.chat_api import router as chat_router

app = FastAPI()

# 配置允许跨域的源，可以根据你前端地址调整
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # 你还可以添加其他允许的域名
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # 允许的前端地址列表
    allow_credentials=True,
    allow_methods=["*"],     # 允许所有方法（GET, POST等）
    allow_headers=["*"],     # 允许所有请求头
)

app.include_router(chat_router)
