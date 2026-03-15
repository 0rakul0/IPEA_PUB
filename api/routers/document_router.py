from fastapi import APIRouter, Depends
from typing import Optional
from api.models.document_models import (
    DocumentListResponse,
    DocumentDetailResponse,
)
from api.services.document_service import DocumentService
from api.dependencies import get_document_service

router = APIRouter()


@router.get(
    "/documents",
    response_model=DocumentDetailResponse | DocumentListResponse,
)
async def list_documents(
    author: Optional[str] = None,
    ano: Optional[int] = None,
    tipo: Optional[str] = None,
    limit: int = 50,
    document_service: DocumentService = Depends(get_document_service),
):
    return document_service.search_documents(
        author=author,
        ano=ano,
        tipo=tipo,
        limit=limit,
    )
