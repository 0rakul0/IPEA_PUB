from pydantic import BaseModel
from typing import List, Optional


class DocumentBase(BaseModel):
    document_id: str
    titulo: Optional[str] = None
    autores: Optional[str] = None


class DocumentDetail(DocumentBase):
    ano: Optional[int] = None
    tipo_conteudo: Optional[str] = None
    link: Optional[str] = None
    link_download: Optional[str] = None


class DocumentListResponse(BaseModel):
    documentos: List[DocumentBase]


class DocumentDetailResponse(BaseModel):
    documentos: List[DocumentDetail]