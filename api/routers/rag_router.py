from fastapi import APIRouter, Depends, HTTPException
from api.models.rag_models import RAGResponse, RAGRequest
from api.services.rag_service import RagService
from api.services.search_service import SearchServiceError
from api.dependencies import get_rag_service

router = APIRouter()

@router.post("/rag", response_model=RAGResponse)
async def rag(
    request: RAGRequest,
    rag_service: RagService = Depends(get_rag_service),
):
    try:
        return rag_service.generate_answer(request.query, request.limit)
    except SearchServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
