from fastapi import APIRouter
from api.models.search_models import SearchResponse, SearchRequest
from api.services.search_service import SearchService
from api.config.settings import settings

router = APIRouter()

search_service = SearchService(
    qdrant_url=settings.qdrant_url,
    qdrant_api_key=settings.qdrant_api_key,
    collection_name=settings.collecion_name,
)

@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    return search_service.search(request.query, request.limit)
