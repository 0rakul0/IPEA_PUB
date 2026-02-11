import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

COLLECTION_NAME = "publicacoes_ipea"

DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
COLBERT_MODEL = "colbert-ir/colbertv2.0"
SPARSE_MODEL = "Qdrant/bm25"

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
)

qdrant.delete_collection(COLLECTION_NAME)

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(size=768, distance=models.Distance.COSINE),
        "colbert": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            )
        ),
    },
    sparse_vectors_config={"sparse": models.SparseVectorParams()}
)
