import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

COLLECTION_NAME = "publicacoes_ipea"

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=120,
)

qdrant.delete_collection(COLLECTION_NAME)

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(size=1024,
                                     distance=models.Distance.COSINE,
                                     on_disk=True),
        "colbert": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            )
        ),
    },
    sparse_vectors_config={"sparse": models.SparseVectorParams()},
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=True
        )
    )
)


fields_to_index = [
    ("metadata.document_id", models.PayloadSchemaType.KEYWORD),
    ("metadata.titulo", models.PayloadSchemaType.KEYWORD),
    ("metadata.ano", models.PayloadSchemaType.INTEGER),
    ("metadata.tipo_conteudo", models.PayloadSchemaType.KEYWORD),
]

for field_name, schema in fields_to_index:
    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=field_name,
        field_schema=schema,
    )
    print(f"Índice criado para {field_name}")