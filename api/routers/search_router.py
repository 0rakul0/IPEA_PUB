from fastapi import APIRouter, Depends, HTTPException
from api.models.search_models import SearchResponse, SearchRequest
from api.services.search_service import SearchService, SearchServiceError
from api.dependencies import get_search_service

router = APIRouter()

@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
):
    try:
        return search_service.search(request.query, request.limit)
    except SearchServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
