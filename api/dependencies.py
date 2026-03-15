from functools import lru_cache

from api.config.settings import settings
from api.services.document_service import DocumentService
from api.services.rag_service import RagService
from api.services.search_service import SearchService


@lru_cache
def get_search_service() -> SearchService:
    return SearchService(
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        collection_name=settings.collecion_name,
    )


@lru_cache
def get_rag_service() -> RagService:
    return RagService(search_service=get_search_service())


@lru_cache
def get_document_service() -> DocumentService:
    return DocumentService(
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        collection_name=settings.collecion_name,
    )
