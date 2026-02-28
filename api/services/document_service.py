from qdrant_client import QdrantClient, models
from api.models.document_models import (
    DocumentBase,
    DocumentDetail,
    DocumentListResponse,
    DocumentDetailResponse,
)


class DocumentService:
    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str):
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name

    def list_documents(self) -> DocumentListResponse:
        documentos_dict = {}
        offset = None

        while True:
            points, offset = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )

            if not points:
                break

            for point in points:
                metadata = point.payload.get("metadata", {})
                document_id = metadata.get("document_id")

                if document_id and document_id not in documentos_dict:
                    documentos_dict[document_id] = {
                        "document_id": document_id,
                        "titulo": metadata.get("titulo"),
                        "autores": metadata.get("autores"),
                    }

            if offset is None:
                break

        return DocumentListResponse(
            documentos=[
                DocumentBase(**doc)
                for doc in documentos_dict.values()
            ]
        )

    def search_documents(
            self,
            author: str | None = None,
            ano: int | None = None,
            tipo: str | None = None,
            titulo: str | None = None,
            document_id: str | None = None,
            limit: int = 50,
    ) -> DocumentDetailResponse | DocumentListResponse:

        # Se não houver filtros → retorna lista simples
        if not any([author, ano, tipo, titulo, document_id]):
            return self.list_documents()

        must_conditions = []

        # TEXT → MatchText
        if author:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.autores",
                    match=models.MatchText(text=author)
                )
            )

        if titulo:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.titulo",
                    match=models.MatchText(text=titulo)
                )
            )

        if tipo:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.tipo_conteudo",
                    match=models.MatchText(text=tipo)
                )
            )

        # INTEGER → MatchValue
        if ano:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.ano",
                    match=models.MatchValue(value=ano)
                )
            )

        # KEYWORD → MatchValue
        if document_id:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.document_id",
                    match=models.MatchValue(value=document_id)
                )
            )

        scroll_filter = models.Filter(must=must_conditions)

        points, _ = self.qdrant.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        documentos = {}

        for p in points:
            metadata = p.payload.get("metadata", {})
            doc_id = metadata.get("document_id")

            if doc_id and doc_id not in documentos:
                documentos[doc_id] = {
                    "document_id": doc_id,
                    "titulo": metadata.get("titulo"),
                    "autores": metadata.get("autores"),
                    "ano": metadata.get("ano"),
                    "tipo_conteudo": metadata.get("tipo_conteudo"),
                    "link": metadata.get("link_pdf"),
                    "link_download": metadata.get("link_download"),
                }

        return DocumentDetailResponse(
            documentos=[
                DocumentDetail(**doc)
                for doc in documentos.values()
            ]
        )