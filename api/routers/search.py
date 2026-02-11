from fastapi import APIRouter
from api.models.search import SearchResponse, SearchRequest
from api.services.search import SearchService
from api.config.settings import settings

router = APIRouter()


search_service = SearchService(
    qdrant_url=settings.qdrant_url,
    collection_name=settings.collecion_name,
)

@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    return search_service.search(request.query, request.limit)
