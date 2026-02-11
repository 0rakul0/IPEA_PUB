from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from api.config.settings import settings

class EmbeddingsService:
    def __init__(self):
        self.dense_model = TextEmbedding(settings.dense_model)
        self.sparse_model = SparseTextEmbedding(settings.sparse_model)
        self.colbert_model = LateInteractionTextEmbedding(settings.colbert_model)

    def embed_query(self, query):
        """ela processa e retorna"""
        dense = list(self.dense_model.query_embed([query]))[0].tolist()
        sparse = list(self.sparse_model.query_embed([query]))[0].as_object()
        colbert = list(self.colbert_model.query_embed([query]))[0].tolist()

        return dense, sparse, colbert

