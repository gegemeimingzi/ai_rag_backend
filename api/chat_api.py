from fastapi import APIRouter, HTTPException
from schemas.chat_schema import ChatRequest, ChatResponse, SourceDocument
from langchain_core.messages import HumanMessage, AIMessage
from memory.redis_memory import get_message_history
from services.chat_service import build_history_text_with_token_limit, qa_chain

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        history = get_message_history(req.session_id)
        last_user_question = None

        for msg in req.messages:
            if msg.role == "user":
                history.add_message(HumanMessage(content=msg.content))
                last_user_question = msg.content
            elif msg.role == "assistant":
                history.add_message(AIMessage(content=msg.content))

        if not last_user_question:
            raise HTTPException(status_code=400, detail="No user question provided.")

        history_text = build_history_text_with_token_limit(history.messages)
        result = qa_chain.invoke({
            "query": last_user_question,
            "chat_history": history_text
        })

        answer = result.get("result", "")
        used_knowledge = result.get("used_knowledge", False)
        top_score = result.get("top_score", None)
        no_use_reason = result.get("no_use_reason", None)
        raw_docs = result.get("source_documents", [])

        source_documents = []
        for item in raw_docs:
            if isinstance(item, dict):
                source_documents.append(SourceDocument(**item))
            elif isinstance(item, str):
                source_documents.append(SourceDocument(content=item))

        history.add_message(AIMessage(content=answer))

        return ChatResponse(
            answer=answer,
            used_knowledge=used_knowledge,
            top_score=top_score,
            no_use_reason=no_use_reason,
            source_documents=source_documents
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
