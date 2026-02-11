from fastapi import APIRouter
from api.models.rag import RAGResponse, RAGRequest
from api.services.rag import RagService

from api.routers.search import search_service

router = APIRouter()

rag_service = RagService(search_service=search_service)

@router.post("/rag", response_model=RAGResponse)
async def rag(request: RAGRequest):
    return rag_service.generate_answer(request.query, request.limit)
