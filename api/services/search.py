from qdrant_client import QdrantClient, models
from sympy.codegen.cxxnodes import using

from api.models.search import SearchResult, SearchResponse
from api.services.embeddings import EmbeddingsService

class SearchService:
    def __init__(self, qdrant_url: str, collection_name: str):
        self.qdrant = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.embeddings_service = EmbeddingsService()

    def search(self, query: str, limit: int = 3) -> SearchResponse:
        query_dense, query_sparse, query_colbert = self.embeddings_service.embed_query(query)
        """
        dense: semantica.
        sparse: macth das palavras chaves
        colbert: junta tudo
        """
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            prefetch=[
                {
                    "prefetch": [
                        {"query": query_dense, "using": "dense", "limit": 20},
                        {"query": query_sparse, "using": "sparse", "limit": 20},
                    ],
                    "query": models.FusionQuery(fusion=models.Fusion.RRF),
                    "limit": 15,
                }
            ],
            query=query_colbert,
            using="colbert",
            limit=limit,
        )
        max_score = max(results.score for results in results.points)
        search_results = [
            SearchResult(
                score=result.score / max_score if max_score > 0 else 0,
                text=result.payload["text"],
                metadata=result.payload["metadata"],
            )
            for result in results.points
        ]
        return SearchResponse(results=search_results)