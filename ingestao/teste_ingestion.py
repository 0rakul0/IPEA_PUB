"""
Script de ingestão de teste no Qdrant.

Objetivo:
- Validar conexão com Qdrant
- Validar upsert em `publicacoes_ipea`
- Inserir pontos sintéticos (sem depender de PDF/Docling/FastEmbed)
"""

import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

COLLECTION_NAME = "publicacoes_ipea"


def _dense_vector(size: int = 512) -> list[float]:
    return [0.001 * (i + 1) for i in range(size)]


def _colbert_multivector(tokens: int = 3, size: int = 128) -> list[list[float]]:
    return [[0.001 * (j + 1) for j in range(size)] for _ in range(tokens)]


def _sparse_vector() -> dict:
    return {"indices": [1, 7, 42], "values": [0.9, 0.6, 0.3]}


def main() -> None:
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ValueError("QDRANT_URL não encontrado no ambiente/.env")

    client = QdrantClient(url=qdrant_url)

    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": _dense_vector(),
            "sparse": _sparse_vector(),
            "colbert": _colbert_multivector(),
        },
        payload={
            "text": "Documento de teste para validar ingestão no Qdrant.",
            "metadata": {
                "id": "teste-ingestao",
                "titulo": "Teste de Ingestão",
                "status_ingestao": "teste",
                "fonte": "script_create_ingestion_teste",
            },
        },
    )

    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=[point],
        batch_size=1,
        wait=True,
    )

    print(f"[OK] Ponto de teste enviado para a coleção '{COLLECTION_NAME}'.")
    print(f"[OK] Point ID: {point.id}")


if __name__ == "__main__":
    main()
