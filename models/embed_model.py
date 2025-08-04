from langchain_huggingface import HuggingFaceEmbeddings

def load_embedding_model(model_name="C:\\Users\\30300\\Desktop\\LLM\\bge-small-zh-v1.5"):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )