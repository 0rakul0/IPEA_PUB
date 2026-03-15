import logging

from qdrant_client import QdrantClient, models
from api.models.search_models import SearchResult, SearchResponse
from api.services.embeddings import EmbeddingsService


logger = logging.getLogger(__name__)


class SearchServiceError(RuntimeError):
    pass


class SearchService:
    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str):
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        self._embeddings_service: EmbeddingsService | None = None

    @property
    def embeddings_service(self) -> EmbeddingsService:
        if self._embeddings_service is None:
            self._embeddings_service = EmbeddingsService()
        return self._embeddings_service

    def search(self, query: str, limit: int = 3) -> SearchResponse:
        try:
            query_dense, query_sparse, query_colbert = self.embeddings_service.embed_query(query)
        except Exception as exc:
            logger.exception("Falha ao gerar embeddings para a consulta %r", query)
            raise SearchServiceError(
                "Falha ao gerar embeddings da consulta. Verifique os modelos e dependencias do fastembed."
            ) from exc
        """
        dense: semantica.
        sparse: macth das palavras chaves
        colbert: junta tudo
        """
        try:
            results = self.qdrant.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    {
                        "prefetch": [
                            {"query": query_dense, "using": "dense", "limit": 20},
                            {"query": query_sparse, "using": "sparse", "limit": 20},
                        ],
                        "query": models.FusionQuery(fusion=models.Fusion.RRF),
                        "limit": 20,
                    }
                ],
                query=query_colbert,
                using="colbert",
                limit=limit,
            )
        except Exception as exc:
            logger.exception(
                "Falha ao consultar o Qdrant na colecao %r para a consulta %r",
                self.collection_name,
                query,
            )
            raise SearchServiceError(
                "Falha ao consultar o Qdrant. Verifique URL, chave da API, colecao e configuracao dos vetores."
            ) from exc
        if not results.points:
            return SearchResponse(results=[])

        max_score = max(result.score for result in results.points)
        search_results = [
            SearchResult(
                score=result.score / max_score if max_score > 0 else 0,
                text=result.payload["text"],
                metadata=result.payload["metadata"],
            )
            for result in results.points
        ]
        return SearchResponse(results=search_results)
