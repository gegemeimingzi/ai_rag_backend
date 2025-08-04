import os
import re
import warnings
from tqdm import tqdm
from langchain.schema import Document
from langchain_chroma import Chroma
from models.embed_model import load_embedding_model



def extract_source_text(text: str) -> str:
    """
    提取法条编号，例如《中华人民共和国宪法》第五十九条 => 第五十九条
    """
    match = re.search(r"第[一二三四五六七八九十百千0-9]+条", text)
    return match.group(0) if match else ""

def load_txt_documents_with_metadata(directory: str):
    """
    加载目录下所有 .txt 文件，并将其按行切分为多个 Document，附带法条编号和文件元信息。
    """
    documents = []
    file_paths = []

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".txt"):
                file_paths.append(os.path.join(root, filename))

    for full_path in tqdm(file_paths, desc="📂 加载法律文本文件"):
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                full_text = f.read()

            file_name = os.path.splitext(os.path.basename(full_path))[0]
            entries = [para.strip() for para in full_text.strip().split("\n") if para.strip()]

            for entry in entries:
                source_text = extract_source_text(entry)
                metadata = {
                    "source_file": full_path,
                    "source_name": file_name,
                    "source_text": source_text
                }
                documents.append(Document(page_content=entry, metadata=metadata))

        except UnicodeDecodeError as e:
            print(f"⚠️ 文件读取失败: {full_path}，错误: {e}")

    return documents


if __name__ == "__main__":
    # ✅ 加载嵌入模型
    embedding = load_embedding_model()

    # ✅ 向量存储路径配置
    persist_dir = r"C:\Users\30300\Desktop\Rag_Project\chroma_persist"
    collection_name = "legal_docs"

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("✅ 检测到已有向量数据库，正在加载...")
        vectordb = Chroma(
            embedding_function=embedding,
            persist_directory=persist_dir,
            collection_name=collection_name
        )
    else:
        print("🚀 未检测到向量库，开始加载文本并构建新库...")
        data_dir = r"C:\Users\30300\Desktop\Rag_Project\data\Chinese-Laws"
        raw_documents = load_txt_documents_with_metadata(data_dir)
        print(f"✅ 成功加载 {len(raw_documents)} 条法律条文")

        vectordb = Chroma.from_documents(
            documents=tqdm(raw_documents, desc="📌 正在生成向量并写入数据库"),
            embedding=embedding,
            persist_directory=persist_dir,
            collection_name=collection_name
        )
        print(f"✅ 向量数据库已构建完毕，共包含 {len(raw_documents)} 条法条片段")

    # ✅ 示例检索
    print("\n📌 示例查询结果：")
    results = vectordb.similarity_search("工会权益", k=3)
    for i, doc in enumerate(results, start=1):
        print(f"\n文档 {i}:")
        print("内容摘要:", doc.page_content[:100].replace("\n", " "))
        print("来源文件:", doc.metadata.get("source_name"))
        print("法条来源:", doc.metadata.get("source_text"))
