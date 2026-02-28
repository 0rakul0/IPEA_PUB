from fastapi import APIRouter
from typing import Optional
from api.models.document_models import (
    DocumentListResponse,
    DocumentDetailResponse,
)
from api.services.document_service import DocumentService
from api.config.settings import settings

router = APIRouter()

document_service = DocumentService(
    qdrant_url=settings.qdrant_url,
    qdrant_api_key=settings.qdrant_api_key,
    collection_name=settings.collecion_name,
)


@router.get(
    "/documents",
    response_model=DocumentDetailResponse | DocumentListResponse,
)
async def list_documents(
    author: Optional[str] = None,
    ano: Optional[int] = None,
    tipo: Optional[str] = None,
    limit: int = 50,
):
    return document_service.search_documents(
        author=author,
        ano=ano,
        tipo=tipo,
        limit=limit,
    )